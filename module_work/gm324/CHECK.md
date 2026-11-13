# GM-324 and GM-325: liquefaction outputs, site response, signals

Branch `claude/gm-324-325-seismic-outputs`, from master 2feea3d. 2026-11-13.
Modules: `liquepy_agent`, `seismic_geotech`, `pystrata_agent`,
`seismic_signals_agent`, and their adapters. Every finding was opened and
confirmed on that master before any change.

## Two things the audit did not say

1. **Bracketed duration was never computed.** The audit's row reads "returns
   0.0 when its calculation fails". It failed every time:
   `intensity_measures.py` called `eqsig.im.calc_brac`, which does not exist
   in eqsig (1.2.17 has `calc_brac_dur(asig, threshold)`), caught the
   `AttributeError` and returned 0.0. Every `bracketed_duration_s` the tool
   has ever returned was that 0.0.
2. **The manual's own liquefaction example moves.** Its inputs have layered
   unit weights (18, 18, 19, 19 kN/m3) and run the default B&I (2014) SPT
   route, so its saved output changes with the total-stress rule (minimum
   FOS 0.435 to 0.426).

## What the print says

| Rule | Printed source |
|---|---|
| Total vertical stress in CSR is the total overburden at the depth | FHWA GEC-13 Vol I Eq. 5-9, CSR = 0.65 (amax/g)(sigma_v/sigma'_v) rd, as the NCEER route already cites it. That it is the sum of gamma x thickness over the layers above is arithmetic; no page is needed for it and none prints the other rule (the depth's own gamma times the whole depth). |
| Mean effective stress under K0 conditions | NAVFAC DM7.2 Eq. 6-22: sigma'm = (1/3)(1 + 2(1 - sin phi')) sigma'v. The code uses the same form with K0 in place of 1 - sin(phi'). K0 = 0.5 is that expression at phi' = 30 deg. |
| The Darendeli and Menq curves depend on the mean effective confining stress | FHWA GEC-5 section 8.8 (p. 8-24): Darendeli (2001) includes confining pressure; the curve shape depends on sigma'm. |
| A bracketed-duration threshold | No held reference prints one. Searched the reference library's text for "bracketed duration": no hit. |
| Robertson and Cabal (2012) permeability from Ic | No held reference prints it. Searched for the relation's constant (0.952): no hit. |

## Findings taken

Rows are the audit's (`module_work/module_audit_2026-11-12/results/seismic.md`).

| Audit row | Sev | What I did |
|---|---|---|
| liquepy 1: CPT tool told to return `n_liquefiable` and other key names | A | **Fixed.** The result now carries `n_liquefiable` (readings below the water table, Ic under the limit, FS below 1.0) and `n_susceptible`. The `cpt_liquefaction` text now names the keys the result has (`lpi`, `lsn`, `ldi_m`, `min_fos`), and describes `max_settlement_mm`, which was returned and described nowhere. |
| liquepy 2: `field_correlations` told to return profiles | A | **Fixed, one part stopped short.** The result carries `profile` (Vs, Dr, su ratio, Ic by depth) and the su-ratio and Ic ranges. Permeability is not returned and is taken out of the tool's brief: see "Stopped short". |
| liquepy 3 and seismic_geotech 8: two total-stress rules | B | **Fixed.** One rule on both routes, stated in the parameter text and in `choices`. |
| liquepy 4: `min_fos` falls back to all readings with no flag | B (same function) | **Stated.** `min_fos_basis` in every result; `warnings` when the fallback is used. |
| seismic_geotech 1: K_sigma not applied, no depth flagged | B (same function) | **Stated.** The K_sigma entry of `choices` names the depths with sigma'v above 100 kPa. |
| seismic_geotech 10: `returns` leaves out keys | C | **Fixed** for `liquefaction_evaluation` only (`layer_results`, `magnitude`), because the new test reads that text. |
| pystrata 1: mean effective stress from total stress, K0 0.5, no water table | A | **Fixed.** `water_table_depth` and `k0` are inputs; the result states both, with each curve layer's stresses. |
| pystrata 2: spectra and profiles computed, not returned | B | **Fixed.** |
| pystrata 3: outcrop motion never said | B (same function) | **Stated**, in the `motion` text and in `choices`. |
| seismic_signals 1: bracketed duration 0.0 on failure, threshold never stated | A | **Fixed.** Computed from a threshold the caller gives; None with the reason otherwise. |

## What changed

### Liquefaction (commit 9dbc4b6)

- `liquepy_agent/spt_liquefaction.py`: total stress summed through the
  layers above, each `gamma` applying from the previous evaluation depth (the
  ground surface for the first) to its own. Depths must increase and be
  positive, as the NCEER route already required of increasing depths.
- `TOTAL_STRESS_RULE`, the same text in `liquepy_agent.spt_liquefaction`
  and `seismic_geotech.liquefaction` (analysis modules do not import each
  other; a test compares the two). Both routes' `choices` carry it.
- `liquepy_agent/cpt_liquefaction.py`, `results.py`: `n_susceptible`,
  `n_liquefiable`, `min_fos_basis`, `warnings`.
- `FieldCorrelationsResult.profile()`: one row per reading up to 40; a
  longer sounding in 40 depth bands of band averages, with `profile_note`.
  A value that is not finite stays missing (None).
- Adapter text: `gamma` on all three liquefaction tools, the CPT and
  field-correlation `returns`.

### pystrata (commit 2670f52)

- `eql_site_response.layer_stress`: sigma_v from the layers' unit weights,
  hydrostatic pore pressure below `water_table_depth` (9.81 kN/m3),
  sigma'm = sigma'v (1 + 2 K0) / 3, floor 5 kPa as before.
- Defaults unchanged: no water table, K0 0.5. With no water table given and
  a computed stress, the result's `warnings` and `choices` say that
  effective stress was taken equal to total stress.
- `to_dict()`: `periods_s`, `Sa_surface_g`, `Sa_input_g` (11 periods, 0.01
  to 5 s), `Sa_surface_max_g`, `period_at_Sa_surface_max_s`,
  `depth_profile`, `water_table_depth_m`, `K0`, `layer_stresses`, `choices`,
  `warnings`. A result with a 20-row profile stays under 8,000 characters.
  (Review fix: the echo is now `k0`, and it and `water_table_depth_m` are
  null with `stress_inputs_note` when no Darendeli or Menq layer used them.)
- **A trap in pystrata's arrays**, found by the new test before it reached a
  result: the strain profile has a leading point at the surface (so
  sub-layer i is entry i + 1 by position), and the two Vs profiles are in
  step-plot form (the first value twice). `depth_profile` matches strain by
  depth and reads Vs from entry i + 1; arrays not of that form give no
  profile instead of a shifted one (review fix: `depth_profile` null with
  the reason in `depth_profile_note`; it was left out without a word).
  Checked against pystrata 0.5.4's `output.py`.

### Signals (commit b1458de)

- `intensity_measures.bracketed_duration`: first to last sample above
  `bracketed_threshold_g`. No threshold, or a record that never exceeds it:
  None, and `bracketed_duration_note` says which.

### Test and manual (commit 63798a3)

- `funhouse_agent/tests/test_gm324_promised_returns.py`: every key a tool's
  `returns` names is in its result, for `cpt_liquefaction`,
  `spt_liquefaction`, `field_correlations`, `liquefaction_analysis`,
  `liquefaction_evaluation`, `eql_site_response`, `linear_site_response` and
  `intensity_measures`. A key present only in some cases is checked in the
  case that shows it.
- `docs/user_manual/manual.html`: 15 lines, the rows of the changed tools
  only. I built the manual in memory and copied across only those lines.
  **Corrected by the review:** this said line 335 differed on master and
  was not this task's. It is this task's: line 335 is the "documented
  parameters" count, 5807 committed against 5812 rebuilt, and the five are
  the inputs this branch adds (`water_table_depth` and `k0` on both
  site-response tools, `bracketed_threshold_g`). On master it rebuilds the
  same. The review fixes carry 5812 (commit ed5b5e8). Only line 1594
  (`gamma_soil`) differs on master, and it was left. Whoever merges another
  branch that adds parameters re-takes line 335.
- `docs/user_manual/worked_examples.json`: the one liquefaction entry.

## Before and after

### One total-stress rule

Made-up profile: depths 2, 5, 8, 12 m; (N1)60 8, 12, 18, 22; fines 5, 10, 15,
8 %; gamma 16.0, 18.5, 19.5, 20.5 kN/m3; water table 3 m; amax 0.25 g; M 7.0.

B&I (2014) SPT, the default route:

| Depth (m) | sigma_v before | sigma_v after | sigma'v before | sigma'v after | FOS before | FOS after |
|---|---|---|---|---|---|---|
| 2 | 32.0 | 32.0 | 32.0 | 32.0 | 0.736 | 0.736 |
| 5 | 92.5 | 87.5 | 72.9 | 67.9 | 0.783 | 0.776 |
| 8 | 156.0 | 146.0 | 106.9 | 96.9 | 1.136 | 1.115 |
| 12 | 246.0 | 228.0 | 157.7 | 139.7 | 1.178 | 1.147 |

NCEER route: unchanged (32.0, 87.5, 146.0, 228.0 kPa before and after). The
two routes now report the same stresses. The effect on FOS is small here
(up to 3 %) because total and effective stress move together; it grows with
the contrast in unit weight and with depth. Where unit weight rises with
depth, as here, the old rule overstated the overburden, and the new FOS is
lower.

### The manual's liquefaction example

Depths 3, 6, 9, 12 m; gamma 18, 18, 19, 19; water table 2 m; 0.3 g; M 7.0.

| Depth (m) | sigma_v before | after | FOS before | after |
|---|---|---|---|---|
| 9 | 171.0 | 165.0 | 0.435 | 0.426 |
| 12 | 228.0 | 222.0 | 0.906 | 0.894 |

Minimum FOS 0.435 to 0.426; four liquefiable layers before and after.

### pystrata confining stress

Made-up profile: 4 m Darendeli (PI 0, 17.5 kN/m3, Vs 160), 8 m Darendeli
(PI 20, 19, Vs 220), 12 m Menq (20, Vs 350) over rock; `synthetic_long`
scaled to 0.1 g.

| Inputs | Mean stress, layers 1 to 3 (kPa) | Surface PGA (g) | Peak strain (%) |
|---|---|---|---|
| before (no inputs existed) | 23.33, 97.33, 228.0 | 0.1949 | 0.0916 |
| after, nothing given | 23.33, 97.33, 228.0 | 0.1949 | 0.0916 |
| after, water table 2 m | 23.33, 58.09, 123.36 | 0.2133 | 0.1189 |
| after, water table 2 m, K0 1.0 | 35.0, 87.14, 185.04 | 0.1959 | 0.0740 |

A call that gives neither input returns the same numbers as before, now
with a warning. With the water table in the profile the confining stress in
this case drops by 40 to 46 % in the layers below it, and the peak strain
rises by 30 %.

### Bracketed duration

`synthetic_pulse`: before, 0.0 s; after, None with the reason when no
threshold is given, 18.27 s above 0.05 g.

## Tests

Run with `/tmp/v/bin/python` (Python 3.12, liquepy 0.6.34, pyStrata 0.5.4,
eqsig 1.2.17, openseespy 3.8.0, numpy 2.5.4), judged by pytest's exit code.

One run over `liquepy_agent`, `seismic_geotech`, `pystrata_agent`,
`seismic_signals_agent`, `opensees_agent`, `calc_package`,
`webapp/tests/test_page_fingerprints.py`, `test_public_text.py`,
`test_docs_currency.py`, and the 27 test files under `funhouse_agent`,
`calc_package`, `webapp/tests`, `validation_examples` and
`foundry_test_harness` that name these modules (all of
`test_defaults_used.py` among them): **2,962 passed, 28 skipped, 1 failed.**

- The one failure is `test_defaults_used.py::...[profile_figure|plot_data]`
  (`logx` and `logy` change nothing in the result). It fails the same way on
  master 2feea3d with my changes stashed, in this environment. Not this
  task's; not fixed.
- The skips: 14 slope_stability probes (the GM-140 hold), 9 for python-ags4
  and pydiggs not installed, 3 for pdflatex, 1 pystrata 0.6 test (Python
  3.13 only), 1 "pystrata is installed" error-path test. None is in a module
  this task changed except the last two.
- New tests: 9 in `liquepy_agent/tests/test_gm324.py`, 17 in
  `pystrata_agent/tests/test_gm325.py`, 6 in
  `seismic_signals_agent/tests/test_bracketed_duration.py`, 10 in
  `funhouse_agent/tests/test_gm324_promised_returns.py`.
- `module_work/doc10/check_manual.py`: OK.
- Fingerprints: `python -m webapp.tests.page_builds record` wrote a file
  identical to the committed one, so there is no fingerprint commit.

## Calls made

1. **No default threshold for bracketed duration.** `seismic_signals_agent/
   DESIGN.md` described it as exceedance of 0.05 g, but no held reference
   prints a threshold and the code never applied one. I made it an input
   with no default. If the owner wants 0.05 g as the default, it is a
   one-line change and needs a page to cite.
2. **pystrata's defaults stay "no water table, K0 0.5".** The brief asked for
   both as inputs with the assumption stated. The alternative was to refuse
   a Darendeli or Menq layer with no water table given, as the liquefaction
   tools refuse. I kept the old numbers and added the warning, because a
   site-response profile can truly have no water table and a refusal would
   stop every existing call. The owner may prefer the refusal.
3. **`n_liquefiable` on a CPT** counts readings that are below the water
   table, have Ic under the limit, and have FS below 1.0. The SPT result
   counts layers with FS below 1.0; the CPT count adds the susceptibility
   test because liquepy gives a factor of safety at every reading, sand or
   clay, wet or dry.
4. **The SPT B&I route now refuses a depth of 0 or less and depths out of
   order.** The layered sum needs ordered depths; the NCEER route already
   refused them out of order.
5. **Reported periods for pystrata**: 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75,
   1, 2, 3, 5 s. The OpenSees tool reports 0.1 to 3 s; I added the ends
   because pystrata computes 0.01 to 10 s.
6. **`k0` above 3 is refused** (review finding 10). No held reference
   prints that bound. It is a guard against a slip (a value in the wrong
   unit or a passive coefficient), set near Kp for phi' 30 deg; a heavily
   overconsolidated clay's K0 can approach it. If the owner wants it wider,
   it is one number in `_validate_stress_inputs`.

## Stopped short

- **Permeability from `field_correlations`.** The value is computed by
  liquepy's `est_permeability_robertson_and_cabal_2012` and stored on the
  result as `permeability_cm_per_s`. The function's docstring gives no unit
  and no held reference prints the relation, so I cannot confirm cm/s. I
  believe the published relation is in m/s, which would make the attribute
  name wrong by a factor of 100, but that is from memory and not from a
  page. I left the attribute as it was, did not return it, and took
  "permeability" out of the tool's brief. To return it, someone needs the
  source in hand.

## Release note

**Liquefaction, B&I (2014) SPT (the default SPT procedure; tools
`liquefaction.liquefaction_analysis` and `liquepy.spt_liquefaction`).** Total
vertical stress is now the sum of unit weight times thickness over the
layers above each depth, the rule the NCEER procedure already used. Before,
it was the depth's own unit weight times the whole depth. A profile with
one unit weight is unchanged. A profile whose unit weight varies with depth
gets different total and effective stresses, CSR, K_sigma and factor of
safety at every depth below the first; in the manual's own example the
minimum factor of safety moves from 0.435 to 0.426. Depths must now be
positive and in increasing order, on both SPT procedures. The SPT `depth`
text now says the depths are where (N1)60 was measured, with a depth at
each change in unit weight: given layer mid-depths, the sum takes each
change half a layer too high. No other calculated value changes: the
NCEER procedure, the CPT procedure, site response with no new input given,
and every intensity measure except bracketed duration return what they did.
Bracketed duration was 0.0 for every record and is now computed only when a
threshold is given.

**Linear site response (`pystrata.linear_site_response`) now reports its
strains.** `max_shear_strain_pct` was 0.0 on every linear run, because
pystrata's linear calculator computes no peak strain. It is now the peak of
pystrata's strain history at each sub-layer's middle, the quantity the
equivalent-linear run reports (on a linear soil the two now agree). The
result echoes the water table and K0 as `water_table_depth_m` and `k0`
(was `K0`) only when a Darendeli or Menq layer used them, else null with
`stress_inputs_note`; `defaults_used` lists them only on those runs too.
A 'linear' layer is not divided, so its strain is the one at its middle
(the text says so; thinner layers resolve the peak in a thick one).

**Field correlations (`liquepy.field_correlations`).** `dr_min`/`dr_max`
are over the sand-like readings (Ic below 2.6) and `su_ratio_min`/
`su_ratio_max` over the clay-like ones, null when there is none, and the
result gives the split as `i_c_limit`. Before, they were over every
reading, so a sand reading could set the su ratio's maximum.

## Noted, not fixed

- `permeability_cm_per_s`: the unit in the name is unchecked (above).
- pystrata: a run can stop before `max_iterations` and still report
  `converged: false` (seen on both made-up profiles: 7 or 8 iterations of
  15). The module's own convergence measure and pystrata's stopping rule
  disagree. Existing behaviour; it does not change a returned value, but a
  result that says "not converged" after pystrata stopped by itself is
  worth a look.
- pystrata's `plot_profile` and `plot_Vs_profile` draw acceleration and Vs
  against the strain profile's depths, which are the wrong depths for them
  (the trap above). No tool reaches the plots.
- `depth_profile` on pystrata 0.6: since the review fixes it has run on
  0.6.0 (Python 3.13 in WSL, the touched test files and every test file
  that names pystrata). 0.6 writes the strain profile with one more point
  (the base of the soil column) and the Vs and acceleration profiles as
  0.5 does. If a later version changes them, the profile is null with the
  reason, not shifted.
- `liquefaction_analysis` accepts `N` and `n_values` as spellings of (N1)60
  (audit liquepy 5): not in the functions this task changed.
- Effective stress at or below zero is set to 1.0 kPa on both SPT routes
  without a word (audit seismic_geotech 2, liquepy 7). It needs a unit
  weight below water's to happen.
- `k0` was listed in `defaults_used` for a profile with no Darendeli or
  Menq layer, where it does nothing (`applies_when` cannot read a key
  inside a list item). **Fixed in the re-check round (R1):** both stress
  inputs are `applies_when: False`, and the module's own report lists
  them exactly when a layer's stress was computed; the two probe-list
  entries that hid it are gone.
- Manual line 1594 differs from a rebuild on master today; not this task's
  row. (Line 335 was this task's: see "Test and manual".)
- `test_defaults_used.py::...[profile_figure|plot_data]` fails on master in
  this environment.

## Review fixes (branch `fix/gm324-review`)

An independent review of 2d028c3 found 12 things (one A, five B, six C).
Commits: 90e638c (pystrata), 18b5d5b (liquefaction), ed5b5e8 (manual),
then this file.

| # | Grade | Finding | What it became |
|---|---|---|---|
| 1 | A | `linear_site_response` gave peak strain 0.0 in the scalar and in every `depth_profile` row: pystrata's linear calculator never sets `strain_max` and the module turned None into 0.0 | **Fixed by computing it.** A linear run adds pystrata's documented `StrainTSOutput` at each sub-layer's middle and takes each history's peak, which is what the equivalent-linear calculator stores as `strain_max` (`motion.calc_peak(calc_strain_tf(input, layer middle))`, pystrata 0.5.4 `propagation.py`). Checked against the closed form for one damped layer on an elastic half-space. A strain pystrata does not give is null, never 0.0. The first build failed CI's Python 3.13 job (pystrata 0.6.0): 0.6's `MaxStrainProfile` adds a point at the base of the soil column, the linear strains were laid out for 0.5, the counts disagreed and the linear `depth_profile` came back null (`test_gm325`: `test_linear_analysis_takes_them_too`, `test_linear_strain_matches_the_closed_form`). Now each of `MaxStrainProfile`'s depths takes the strain of the sub-layer it lies in, on either version; run on 0.5.4 and on 0.6.0 (Python 3.13 in WSL). |
| 2 | B | SPT `depth` said "layer mid-depths" while `gamma` sums from the previous depth down | **Fixed (text).** `depth` on `liquefaction_analysis`, `spt_liquefaction` and `liquefaction_evaluation`, and both docstrings: the depths are where (N1)60 was measured; give a depth at each change in unit weight (a layer's base, the water table where the unit weight changes there), or the change is taken at the depth above. The result's total-stress `choices` basis says the same (both copies of `TOTAL_STRESS_RULE`). No arithmetic change. |
| 3 | B | Dr and su/sigma'v ranges over every reading | **Fixed.** Dr over Ic below `i_c_limit` (2.6), su/sigma'v over Ic at or above it, null when there is none; the `returns` text says so. The profile rows still carry every correlation at every reading (the profile text says to read them by Ic): not changed. |
| 4 | B | `water_table_depth_m` and `K0` echoed as used on profiles that did not use them; `K0` hid k0 from the GM-160 guard | **Fixed.** Echoed only when a Darendeli or Menq layer had its stress computed; else null with `stress_inputs_note`. Echo renamed `k0`. (This round first added `NOT_REACHED` and `ECHO_IS_NOT_THE_INPUT` entries for the two pystrata probes; the re-check showed they hid `defaults_used` listing k0 where it was unused, and R1 below replaced them.) |
| 5 | B | CHECK.md wrong about manual line 335 | **Fixed.** Line 335 carries 5812; CHECK.md corrected ("Test and manual"). |
| 6 | B | `depth_profile` left out with no word when pystrata's arrays are not in the expected form | **Fixed.** `depth_profile` null and `depth_profile_note` says which check failed. Merge after the Python 3.13 CI job (pystrata 0.6) is green. |
| 7 | C | NCEER route took a first depth of 0 and returned FOS 99.9 | **Fixed.** Refused with the reason, as on B&I. |
| 8 | C | Both SPT routes flag layers above the water table `liquefiable` | **Not fixed**: it changes saved output. Under "Follow-ups". |
| 9 | C | The unified tool's `returns` left out `warnings` and five CPT keys | **Fixed.** Listed as "CPT only"; the promised-returns test now runs a dry CPT case to see `warnings`. |
| 10 | C | `k0` above 3 refused with no printed basis | **Listed** under "Calls made" (6). |
| 11 | C | The foundry wrappers describe the old results | **Not fixed**: INF-06 deletes `foundry/`. |
| 12 | C | The bracketed threshold is checked after the other measures are computed | **Not fixed**: optional (the review's word); it changes no value, only when an invalid threshold is refused. |

### Before and after (made-up inputs)

| Case | Before (2d028c3) | After |
|---|---|---|
| 1: method_probes' linear profile (10 m, Vs 200, linear) through `linear_site_response` | `max_shear_strain_pct` 0.0, row 0.0 | 0.0352, row 0.0352 (`eql_site_response` on the same input: 0.0352; PGA 0.3022 both) |
| 1: one 10 m linear layer (Vs 200, 18 kN/m3, 2 %) on rock (Vs 760, 22, 1 %), `synthetic_pulse`, strain at 5 m | 0.0 | 0.046341 % (closed form: 0.046341 %) |
| 4: linear profile with water table 2.0 and k0 0.7 | `water_table_depth_m` 2.0, `K0` 0.7, no note | both null, `stress_inputs_note` says no Darendeli or Menq layer used them |
| 3: CPT sand / clay band / sand (water 2.4 m) | Dr -0.193 to 0.554; su/sigma'v 0.617 to 11.83 | Dr 0.328 to 0.554; su/sigma'v 0.617 to 0.858 |
| 2: 0-4 m at 16 kN/m3 over 20, water 1 m, depths 2 and 7 m | sigma_v at 7 m 132.0 kPa (true 124.0 with a depth at 4 m; FS 0.420 against 0.402) | the same numbers by design; the depth text and `choices` now say to give the 4 m depth |
| 7: NCEER, first depth 0 | FOS 99.9 for that row | refused: "layer_depths must be positive" |
| 5: manual line 335 | 5807 | 5812 |

### Tests (fix branch)

`C:\dev\GeotechStaffEngineer\.venv` (Python 3.14, pystrata 0.5.4), the
worktree first on the path, one file at a time, judged by exit code; all 0:
`pystrata_agent/tests/test_gm325.py` 21 passed;
`pystrata_agent/tests/test_pystrata_agent.py` 62 passed, 1 skipped;
`funhouse_agent/tests/test_gm324_promised_returns.py` 12 passed;
`liquepy_agent/tests/test_gm324.py` 10; `liquepy_agent/tests/test_choices.py`
6; `liquepy_agent/tests/test_liquepy_agent.py` 59;
`seismic_geotech/tests/test_choices.py` 6;
`seismic_geotech/tests/test_seismic_geotech.py` 84;
`funhouse_agent/tests/test_module_followup_adapters.py` 48;
`funhouse_agent/tests/test_defaults_used.py -k "pystrata or allow_lists or
liquef or liquepy or seismic_geotech"` 53 (its first run failed 2: the
pystrata probes' defaults list k0 = 0.5 and the result now echoes null;
fixed with the `ECHO_IS_NOT_THE_INPUT` entries);
`funhouse_agent/tests/test_adapter_ergonomics.py` 170;
`webapp/tests/test_public_text.py` 5. `check_manual.py --rebuild`: only line
1594 differs, as on master. Page fingerprints carry no tool text: no
re-record.

### Follow-ups that change a saved report's output

- Review finding 8: both SPT routes mark layers above the water table
  `liquefiable` and count them in `n_liquefiable` (p8: layers at 1 and 2 m
  over a 3 m water table, FS 0.52 and 0.515, both True). The CPT count
  already excludes them. For the backlog.
- Re-check C-a: pystrata divides only layers with curves, so a 'linear'
  layer has one strain, at its middle, and the peak inside a thick one is
  missed (the re-check's made-up two-layer case: 0.0283 % with 'linear'
  layers, 0.0497 % with the same soils divided). The tool text now says
  so; dividing linear layers in the module would change saved strains.
  For the backlog if wanted.

## Re-check fixes

An independent re-check of 0f7f717 closed A1, B2, B3, B5, B6, C7 and C9
with its own numbers, confirmed the Python 3.13 fix from pystrata 0.6's
source, and asked for one fix and four small ones.

| # | Grade | Finding | What it became |
|---|---|---|---|
| R1 | B | `defaults_used` listed `k0` 0.5 and the "no water table" case on runs whose result said neither was used (a linear-only profile, or every curve layer with its own `stress_mean`); the two probe-list entries hid it from the GM-158 and GM-160 guards | **Fixed.** `applies_when: False` on `water_table_depth` and `k0` in both pystrata methods, with a comment: `stress_choices` reports both as "default" exactly when a layer's stress was computed, and `_defaults_used` keeps a reported name. Both probe-list entries deleted (`method_probes.py` is master's again). New test through the tool: a Darendeli run that computes the stress lists both; a linear-only run and a `stress_mean` run list neither. |
| C-a | C | A 'linear' layer is not divided, so its strain is the one at its middle | **Stated** in `max_shear_strain_pct` (linear) and `depth_profile` (both methods), and in DESIGN.md; dividing is a follow-up (above). |
| C-b | C | The ranges' split was in the `returns` text only | **Fixed.** `field_correlations` returns `i_c_limit`; listed in `returns` and the manual row; tested with a moved limit. |
| C-c | C | The notes named Python 3.12 for the local runs | **Fixed.** The Windows venv is Python 3.14 (CHECK.md "Tests (fix branch)", pystrata DESIGN.md). |
| C-d | C | `max_shear_strain_pct`'s text said "the largest max_strain_pct of depth_profile", though it is given when the profile is null | **Fixed.** It is the largest of pystrata's sub-layer strains, and the text says it is still given when `depth_profile` is null. |
