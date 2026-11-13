# GM-326 and GM-327: site data and drawing import

Branch `claude/gm-326-327-subsurface-dxf`, from master 2feea3d, 2026-11-13.
Brief: `module_work/cloud_briefs/AUDIT_STAGE2.md`, section "GM-326 and
GM-327". Audit rows: `module_work/module_audit_2026-11-12/AUDIT.md` section 4,
`subsurface_characterization` and `dxf_import` (scout file
`results/site_data.md`).

Every finding below was opened and confirmed on master before a change. One
finding turned out worse than the audit says (the GEF reader, section 2).

## 1. What the print says

Neither module has a held printed reference behind the points fixed here, and
none was invented.

- **Trend fits.** No formula changed. The rules applied are arithmetic: a
  line needs two points; the logarithm exists only for positive numbers; a
  line through exactly two points has no residual scatter.
  `subsurface_characterization/DESIGN.md` cites Phoon & Kulhawy (1999) for the
  trend and COV; that paper is not in geotech-references and nothing was
  taken from it.
- **GEF void values.** A GEF file states its own missing-value marker per
  column (`#COLUMNVOID= <column>, <value>`); the reader uses the file's own
  statement, as pygef parses it (`CPTData.column_void_mapping`). The GEF
  format specification is not a held reference.
- **DXF units.** The `$INSUNITS` codes for in, ft, mm, cm and m were already
  in `planlens.dxf.units` (1, 2, 4, 5, 6). This branch adds names for the
  other codes of the DXF reference's `$INSUNITS` table (3 miles, 7
  kilometres, ... 21 to 24 US survey units), used only to name the unit in a
  refusal message; no conversion is made from them. The DXF reference is not
  a held print: the names are from the public Autodesk table and should be
  checked by whoever next holds it.
- **Nail properties.** The five values `build_slope_geometry` fills in (bar
  25 mm, hole 150 mm, fy 420 MPa, bond 100 kPa, spacing 1.5 m) are
  `slope_stability.nails.SoilNail`'s own defaults, which the slope tool
  declares for its `nails`. They were not checked against GEC-7 here
  (`slope_stability/` is on hold); this branch only lists them.

## 2. Findings taken

| Audit row | Sev | Finding | What was done |
|---|---|---|---|
| subsurface #1 | A | `compute_trend` fits any `trend_type` but "log_linear" as a line and labels it as asked | **Not done, by the brief:** the dispatch task's (GM-317, branch `claude/gm-316-317-warnings-allowed-values`). Confirmed still true (statistics.py, the `else` branch). |
| subsurface #2 | A | Fewer than 2 points returns slope 0, R-squared 0, COV 0 | **Refused.** `compute_trend` raises `ValueError` (log-linear: fewer than 2 points with depth > 0 and value > 0). A group `compute_grouped_trends` cannot fit is kept with `fitted=False`, `not_fitted_reason`, and null fit values in `to_dict`; `plot_parameter_vs_depth` names it in `skipped` (it was dropped without a word), and says when a trend was asked for and fewer than 2 points were drawn. Unequal `depths` and `values` lengths are refused too. |
| subsurface #2 (2 points) | A | With exactly 2 points: R-squared 1, residual 0, COV 0 | **Stated.** `std_residual` and `cov` are null in `to_dict`, with a note (`scatter_defined=False`); R-squared stays 1 and the note says it is 1 by construction. |
| subsurface #3 | A | Log fit drops points with depth <= 0 or value <= 0 and reports the full count | **Fixed.** `n_points` is the number the fit used; `n_dropped` and `dropped_reason` say what was left out; `to_dict` adds a note. The 0.01 m clamp in `predict` is noted, not fixed. |
| (found here) | | A null or NaN among `compute_trend`'s depths or values made every fitted number NaN (reachable once `parse_cpt` returns nulls) | **Fixed** in the branch's third commit: the point is left out and counted in `n_dropped`, with the count in the note; fewer than 2 complete points is refused. |
| subsurface #4 | A | GEF reader turns a null or non-finite qc, fs, u2, Rf into 0.0, uncounted | **Fixed, and found worse.** See below. |
| subsurface #7 | B | `plot_parameter_vs_depth` does not tell the agent about `skipped` | Already told on master (`returns.skipped`); the text now also names a trend line not fitted. |
| dxf_import #1 | A | Text annotations not converted from drawing units | **Fixed.** Converted like the geometry (parser.py, the annotations loop); `to_dict` rounds them. |
| dxf_import #2 | A | Soil properties default silently (gamma 18, phi 0, c 0, cu 0, E 30000, nu 0.3) | **Refused / stated.** In the agent adapter (`soil_assignments`): a missing unit weight, strength for the layer's case, or (FEM) E and nu is refused in one message naming every gap; what may stand in is listed in `defaults_used` per layer. The PDF import adapter builds the same assignments and now shares the rule. |
| dxf_import #3 | A | Nails get five structural defaults silently | **Stated.** Each of the five not given in `nail_defaults` is listed in `defaults_used` (`nail_defaults.<input>`). The leftmost-point-is-the-head rule is noted, not fixed. |
| dxf_import #6 | B | `units` defaults to "m", never compared with `$INSUNITS` | **Refused / converted, said either way.** No default any more; see "Calls made" 1. |
| dxf_import #7 | B | A missing annotation layer and non-LINE nail entities are skipped silently | **Stated** (same function as #1): both are named in `warnings`, the nail entities with a count by type. |
| dxf_import #8 | C | `parse_geometry` returns keys METHOD_INFO does not list | Listed (`text_annotations`, `units_used`, `units_source`, `header_units`), since the text was being rewritten. |
| dxf_import #4, #5, #9 | B, B, C | Invented layer bottoms (5 m, 2 m); points of a layer merged and sorted by x; the top layer picked by elimination | **Not done.** All three are inside `build_slope_geometry`, `build_fem_inputs` or `_extract_polyline_coords`, which branch `claude/gm-140-step1-one-material-per-point` is changing. No fix here needed one of them: the nail values were made listable from a new constant beside the function (`converter.NAIL_PROPERTY_DEFAULTS`), which a test holds to what the function fills in. |

### The GEF reader: what was really happening

`pygef.read_cpt` defaults to `replace_column_voids=True`, which replaces a
void reading by a value **interpolated from its neighbours** (pygef
`gef/gef.py`, `replace_column_void`: void to null, then `.interpolate()`).
Only a void at the very start or end of a column stays null, and that is the
null the module then turned into 0.0. So on master a missing cone resistance
in the middle of a sounding came back as an invented reading with nothing
said, and one at either end as a measured zero.

The reader now calls pygef with `replace_column_voids=False` and makes each
column's own void value NaN itself. A pygef too old to take the switch (the
pin is `pygef>=0.10`; 0.14.1 was tested) keeps its own behaviour and the
result says, in `warnings`, that interpolated readings cannot be told from
measured ones. The friction ratio pygef computes as fs/qc is made missing
wherever qc or fs is (pygef computes it from the void numbers: -83325 % on
the test file).

**What the readers of the result do with a missing value** (the brief's
question), each checked and tested:

| Reader | Before | Now |
|---|---|---|
| `CPTParseResult.to_dict` (the `parse_cpt` tool) | a number (interpolated or 0.0) | `null` in the array; `missing_values` gives the count per column; a `warnings` line says it |
| `CPTParseResult.summary` | nothing | names the counts |
| `load_cpt_to_investigation` | a measurement of the invented value | no measurement at that depth for that parameter |
| `to_liquepy_inputs` | passed the invented value on | the depth is left out of every array; `rows_dropped` and the reason say so; a column the file does not have at all is still zeros, now named in `columns_assumed_zero` |
| `liquefaction.liquefaction_analysis`, `liquepy.cpt_liquefaction`, `liquepy.field_correlations`, given `parse_cpt`'s arrays by the agent | ran on the invented value | **refused**, naming the column, the count and the first index. Without this the null made every result NaN and the tool answered `min_fos 2.0, lpi 0.0` (no liquefaction) for the whole sounding, which is why the guard was added in `liquepy_agent` although that module is another task's. It is one helper (`refuse_missing_readings`) and one call in each of the two validators. |
| `plot_qc`, `plot_all` (matplotlib) | drew the invented value | a gap in the line (matplotlib's own handling of NaN); not changed |

## 3. Before and after (made-up cases)

**Trend, one point** (`depths=[3.0]`, `values=[12.0]`):
before `{"n_points": 1, "slope": 0.0, "intercept": 0.0, "r_squared": 0.0, "cov": 0.0}`;
after `ValueError: A trend line needs at least 2 points; 1 given. No trend is fitted.`

**Trend, log-linear, 6 points of which 2 cannot enter**
(`depths=[0, 1, 2, 4, 8, 3]`, `values=[5, 10, 20, 40, 80, -1]`):
before `n_points 6`, slope 1.0, R-squared 1.0;
after `n_points 4`, `n_dropped 2`, slope 1.0, R-squared 1.0, and the note
"2 of 6 points left out of the fit: ...". The fitted numbers do not move.

**Trend with a null reading** (`depths=[1, 2, 3, 4]`, `values=[10, null, 30, 40]`):
before, slope, intercept and R-squared all NaN (null in the JSON);
after `n_points 3`, `n_dropped 1`, slope 10.0, and the note "1 of 4 points
left out of the fit: 1 with a missing (null or non-finite) depth or value".

**Trend, two points** (`[1, 3]`, `[10, 20]`): before `std_residual 0.0, cov 0.0`;
after `std_residual null, cov null` with the note. Slope 5.0, intercept 5.0
and R-squared 1.0 unchanged.

**GEF** (`subsurface_characterization/tests/sample_cpt_void.gef`: the sample
sounding with qc at 1.5 m and fs at 3.0 m set to the void value -9999, and Rf
void on both rows):

| Reading | The file | Before | After |
|---|---|---|---|
| qc at 1.5 m | void | 10000 kPa (interpolated from 8000 and 12000) | null |
| fs at 3.0 m | void | 125 kPa (interpolated from 150 and 100) | null |
| Rf at 1.5 m and 3.0 m | void | 1.0 % | null |
| `missing_values` | | not reported | `{"q_c_kPa": 1, "f_s_kPa": 1, "Rf_pct": 2}` |
| measurements in the site model | | 30 | 26 |
| liquefaction on the parsed arrays | | ran on the interpolated profile | refused: "q_c has 1 missing or non-finite reading(s) of 10 (the first at index 2) ..." |

**DXF in millimetres** (a made-up bench 20 m by 10 m drawn in mm, header
`$INSUNITS = 4`, one label), `parse_geometry` with `units` left out:

| | Before | After |
|---|---|---|
| last surface point | (20000.0, 4000.0) "m" | (20.0, 4.0) m |
| label position | (5000.0, 7000.0) | (5.0, 7.0) |
| `units_used` | "m" | "mm", `units_source` "the drawing's header ($INSUNITS)" |

With `units="mm"` given, before: geometry in metres and the label still at
(5000, 7000); after: both in metres. With `units="m"` given against the mm
header: before, the 1000-times section; after, refused, naming both.

**`build_slope_geometry` with a layer `{"name": "Surface"}` and a layer
`{"name": "Clay", "phi": 25}`:** before, a geometry with gamma 18.0, phi 0.0
and 25, c' 0.0 for both, and a nail at 25 / 150 / 420 / 100 / 1.5, nothing
said; after, `ValueError: build_slope_geometry: soil_properties[Surface] has
no gamma, phi; soil_properties[Clay] has no gamma. A drawing gives geometry
only ...`. With gamma and phi given for both, the same geometry as before
and

```
"defaults_used": {
  "soil_properties[Surface].analysis_mode": "drained",
  "soil_properties[Surface].c_prime": 0.0,
  "soil_properties[Surface].gamma_sat": 18.0,
  "soil_properties[Clay].analysis_mode": "drained",
  "soil_properties[Clay].c_prime": 0.0,
  "soil_properties[Clay].gamma_sat": 19.0,
  "nail_defaults.bar_diameter": 25.0, "nail_defaults.drill_hole_diameter": 150.0,
  "nail_defaults.fy": 420.0, "nail_defaults.bond_stress": 100.0,
  "nail_defaults.spacing_h": 1.5}
```

**`build_fem_inputs`** with the same two layers: before, both at gamma 18.0,
E 30000, nu 0.3; after, refused, naming gamma, E and nu for both.

## 4. Calls made (for the owner to revisit)

1. **Drawing units have no default.** Left out, the header's units are used
   and said. A drawing whose header states no units, with none given, is
   **refused** (the other choice was to keep metres and say so). Units that
   differ from the header are **refused** unless `ignore_header_units=true`;
   the other choice was to take the caller's with a warning. Refusal was
   chosen because the agent had been told "default m" and passes "m" by
   habit, and a 1000-times section is the cost; the override exists because
   headers left at a template's value are common. `geo_project.from_dxf` and
   the model-setup tool pass both inputs through.
2. **Which soil inputs are refused and which may stand in.** Refused: gamma;
   phi for a drained slope layer; cu for an undrained one; for FEM gamma, E,
   nu, and phi for a layer with a strength model. Stand in, listed: c' 0,
   `analysis_mode` drained, gamma_sat = gamma; FEM c 0, psi 0, model mc. A
   cohesionless layer given by phi alone is ordinary practice, which is why
   c' 0 is listed and not refused. An undrained clay in FEM must now say
   phi 0 itself.
3. **Nail properties are listed, not refused,** the grout-ground bond stress
   included, although it depends on the ground. They are the slope tool's own
   declared nail defaults and `slope_stability/` is on hold; refusing here
   while the slope tool itself assumes them would be two rules for one input.
   The tool text says to give the bond stress. If GM-140's review makes the
   slope tool refuse a missing bond stress, this should follow.
4. **`defaults_used` on `build_slope_geometry` breaks the straight hand-on.**
   The tool's result is `slope_stability.analyze_slope` input as it comes; a
   `defaults_used` key in it is refused there as an unknown parameter. The
   key is present only when something stood in, and the tool text says to
   take it out. The other choice was to leave the stand-ins unsaid on the one
   tool whose output feeds an analysis. With every input given (gamma_sat and
   analysis_mode included) the result still passes straight through.
5. **Missing CPT readings are null, never interpolated,** which overrides
   pygef's default. An engineer who wants interpolation across a gap can do
   it knowingly from the count. `to_liquepy_inputs` leaves the depth out; the
   liquefaction tools refuse a null.
6. **`compute_trend` refuses; `compute_grouped_trends` flags.** One call
   with too few points is an error; one thin group among several should not
   lose the others, so it comes back marked `fitted=False`.
7. **The same soil-property rule on the PDF import adapter**
   (`pdf_import.build_slope_geometry`, `build_fem_inputs`), which is not one
   of the two modules named. It builds the same assignment objects with the
   same silent values through the same shared check, so leaving it would have
   kept the finding alive one tool over.

## 5. Tests

Run with the brief's venv (Python 3.12, planlens 0.15.1, pygef 0.14.1,
ezdxf 1.4.5, liquepy installed; pydiggs and python-ags4 not installed, their
tests skip and say so).

| Run | Result |
|---|---|
| `funhouse_agent webapp/tests subsurface_characterization dxf_import dxf_export geo_project liquepy_agent foundry_test_harness` (one run, `-n 4`, on commit 40f0282) | 7411 passed, 75 skipped, **2 failed** |
| the two failures, run on master (2feea3d) in the same venv | both fail there too: `test_calc_package_figure_conventions.py::test_large_pile_group_leaves_the_forces_to_the_table` and `test_defaults_used.py::...[profile_figure\|plot_data]`. Neither touches these modules. |
| after the last commit: `subsurface_characterization funhouse_agent/tests/test_subsurface_adapter.py foundry_test_harness webapp/tests/test_public_text.py webapp/tests/test_page_fingerprints.py` | 455 passed, 29 skipped, 0 failed |
| `funhouse_agent/tests/test_defaults_used.py -k subsurface` after the last commit | passed (0 failed) |
| `webapp/tests/test_public_text.py` before each commit | 5 passed |
| `webapp/tests/test_page_fingerprints.py` | 13 passed, before and after a re-record that changed nothing |
| `validation_examples` | not run: no example uses either module (grep) |

Every result was read from pytest's own exit code.

New tests: `subsurface_characterization/tests/test_statistics.py` (refusals,
dropped count, 2-point fit, unfitted group, the plot's `skipped`),
`test_formats_gef.py` (`TestMissingReadingsInAResult`, `TestCPTVoidReadings`
on the new made-up fixture `sample_cpt_void.gef`),
`liquepy_agent/tests/test_liquepy_agent.py` (two refusals),
`dxf_import/tests/test_units_and_defaults.py` (23 tests as built; the
review round in section 8 adds 9, so 32: the five units
cases, annotations, the nail layer, the slope and FEM refusals and lists, the
nail constant held to the function, the agent tool).

Changed tests, each because it relied on a silent value:
`test_statistics.py::test_edge_case_single_point` (expected the zero result);
`test_unknown_positions.py` (one more `skipped` line: the trend not fitted);
`dxf_import/tests/test_integration.py` (takes `defaults_used` out before
`analyze_slope`, and checks it); `funhouse_agent/tests/test_new_adapters.py`
and `test_nested_keys.py` (an FEM layer without nu);
`geo_project/tests/test_ingest.py` (units "mm" against an ezdxf header that
says m). Probes: `dxf_import|build_fem_inputs` now gives nu;
`dxf_import|build_slope_geometry` leaves c' out so the list is exercised.

`funhouse_agent/tests/test_defaults_used.py`: the two guards that give a
run's `defaults_used` back now put a per-item default
(`soil_properties[Clay].c_prime`, `nail_defaults.fy`) into its item
(`_given_back`); for those the value carried into the output is the effect,
so only `defaults_used` itself is set aside when checking that the result
moves. `method_probes.NOT_REACHED` gains `parse_geometry`'s
`ignore_header_units` (the probe gives no units).

Fingerprints: `webapp/tests/test_page_fingerprints.py` passes unchanged, and
`python -m webapp.tests.page_builds record` writes the same file, so there is
no fingerprint commit. Manual: the rows of `compute_trend`, `parse_cpt`,
`dxf_import`'s three tools and `pdf_import`'s two build tools in
`docs/user_manual/manual.html`, rendered with the builder's own row
functions for those methods only (11 lines; no rebuild);
`module_work/doc10/check_manual.py` says OK.

## 6. Release note

**Site data and drawing import say what they did.** A trend fit with fewer
than two points is refused instead of returning a flat line; a log-linear fit
reports how many points it could not use, and `n_points` is now the number it
used. A CPT read from a GEF file keeps a missing reading missing: it used to
come back interpolated from its neighbours, or as zero at the ends of a
column, so cone resistances, sleeve frictions and friction ratios at those
depths change from a number to null, the count is reported, and a
liquefaction run on a profile with a null reading is refused. A DXF is read
in its own header's units unless told otherwise, so a drawing in millimetres
no longer comes out 1000 times too large when `units` is left out; units that
contradict the header are refused unless `ignore_header_units` is set, and a
drawing with no stated units needs `units`. Text labels are in metres like
the geometry. `build_slope_geometry` and `build_fem_inputs` (from a DXF or a
PDF) no longer assume a unit weight of 18, a friction angle of 0, E of 30000
or Poisson's ratio of 0.3: a layer without them is refused, and what does
stand in (c' 0, drained, the five nail properties) is listed in
`defaults_used`. A saved conversation that relied on any of those silent
values gets a refusal that names the input to give.

## 7. Noted, not fixed

- **BRO-XML void readings.** pygef gives no void mapping for a BRO-XML CPT
  (`column_void_mapping` is empty there), so a BRO missing-value marker is
  read as a number. No BRO catalogue is held to cite the marker from. This
  one would change a value in a saved report's output (a BRO sounding with a
  gap): worth an item.
- `TrendAnalysisResult.predict` clamps a depth <= 0 to 0.01 m for a
  log-linear trend when drawing the line (audit #3, second half).
- The unknown `trend_type` (audit #1): GM-317.
- `SoilPropertyAssignment` and `FEMSoilPropertyAssignment` keep their field
  defaults (gamma 18, E 30000, nu 0.3) for a library caller; the agent's
  route no longer reaches them. Removing them belongs with the functions the
  GM-140 branch is editing.
- `foundry/dxf_import_agent_foundry.py` and
  `foundry/subsurface_char_agent_foundry.py` (the retired wrappers) still
  build assignments with the silent values; their trend method now returns
  the flagged group dicts.
- A soil_properties entry with no `name` is still read as "" (it then fails
  to match a boundary, which raises; a single-layer drawing does not need
  the name).
- The nail head is the leftmost end of the line (audit #3, second half).
- The SPLINE flattening tolerance is 0.01 in drawing units whatever the
  units (inside `_extract_polyline_coords`).
- Audit subsurface #5, #6, #8 to #12 and dxf_import #10, #11: outside this
  section.
- `funhouse_agent/tests/test_defaults_used.py::test_every_default_a_run_lists_changes_its_result[profile_figure|plot_data]`
  fails on master in this environment too (logx, logy "changed nothing");
  not this branch's.

## 8. Review round (2026-11-13)

An independent review found B1 to B4 and C1 to C6; the lead ruled on each.
C4 (pygef fallback by signature) and C6 (scope, no change) are not this
round. Branch `fix/gm326-review`.

**Master merged first** (merge commit d7e9775). One conflict:
`docs/user_manual/manual.html` (generated), taken from the rebuild;
`check_manual.py` exit 0 (only the old soe line with `* D_embed` still
differs from a rebuild, as before). The merge alone broke nothing in the
test loop below.

| Finding | Before | After |
|---|---|---|
| B1 units from the header alone | Used with no warning; a drawing in feet under an inch header (`in1_coords_ft.dxf`) came out 12 times too small, said only in `units_source`. | Same numbers, plus a warning: "units was left out, so the drawing's header decided them: $INSUNITS = 1, inches. Read in those units, the section is W m wide and H m high. A header left at a template's value is common (...): check that size against the drawing and give units to confirm or correct it." `parse_geometry`'s `units` text says the same. geo_project's DXF ingest carries it as an `ingest.warning` assumption. |
| B2 phi 0, cohesion left out | c (FEM) or c_prime (slope) 0 stood in and was listed: a layer with no strength. | Refused: "soil_properties[Clay] has phi 0 and no c_prime, so it would have no strength: give c_prime, or analysis_mode='undrained' with cu for an undrained clay" (FEM: "give c (cu for an undrained clay)"). c 0 given with phi 0 runs, with the warning "has phi 0 and c_prime 0 as given: the layer has no strength". pdf_import's two build tools share `soil_assignments`, so they do the same; their soil_properties text now says so too. |
| B3 US survey units | $INSUNITS 21 refused with units left out and again with 'ft', whose message advised the refused call. | 21 and 22 converted (1200/3937 m and 100/3937 m; `units_used` 'us_survey_ft' / 'us_survey_in'); 'ft' or 'in' agrees with them and `units_source` says the survey unit was read, 2 ppm from the international one. Another unit against 21 is refused, and its advice (leave units out) now runs. 23 and 24 stay refused; both refusals point to units with `ignore_header_units=True`, a call that runs. |
| B4 handing the slope result on | Text: "the whole result is analyze_slope input"; `defaults_used` refused there. | The result is flat, so the text names the keys to pass (surface_points, soil_layers, gwt_points, nails) and calls defaults_used and warnings a record, not inputs. slope_stability untouched. `test_the_keys_the_text_names_run_in_analyze_slope` hands them on by dispatch and gets a FOS. The slope part was not nested: that would change the output shape the pdf_import tools and existing tests read. |
| C1 2-point note | "R-squared is 1 by construction", while equal values report 0. | "a line through 2 points has no scatter, so R-squared says nothing about the fit"; values unchanged. |
| C2 unfitted group | `n_points` the count given, `n_dropped` 0. | `n_points` 0, `n_dropped` the count given, as documented; the "left out of the fit" note is for fitted results only (the "Not fitted" note says why). |
| C3 unitless refusal | "$INSUNITS is unitless". | "unitless or absent", plus "A hint only: the header's $MEASUREMENT is 0, English (feet or inches); it does not say which unit." when the header has it. |
| C5 | Section 5 said 27 tests. | 23 as built (corrected above). |

Tests changed by the rulings: `test_statistics.py::test_grouped_trends_flag_the_group`
(n_points 0, n_dropped 1), `::test_group_by_uscs` (every group has
n_points + n_dropped > 0, and n_points > 0 when fitted),
`::test_two_points_say_scatter_is_not_defined` (gains the equal-values
case). New: nine tests in `TestReviewRound` of
`dxf_import/tests/test_units_and_defaults.py`.

Checks: the test loop (the branch's own test files against master, plus
test_probe_warnings, test_allowed_values, test_adapter_ergonomics,
test_defaults_used -k "dxf or subsurface or geo_project or pdf",
test_page_fingerprints, test_public_text) all pass; `check_manual.py` exit
0 with the rebuilt rows copied in. The reviewer's probes, rerun on this
head: 84 of 116 cases unchanged; the rest are the findings above (B1
warning, B2 refusals of the phi-only and FEM no-c cases, B3 survey cases,
C3 wording), master's liquefaction fields (GM-324; WE_liq_spt min_fos 0.435
to 0.426 comes from master, not this round), and the discover_layers file
path (the probe was given a relative path).

Calls made: about 45 tool calls.
