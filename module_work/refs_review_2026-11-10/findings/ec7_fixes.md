# ec7 fix table (branch `fix/review-ec7`, worktree `C:/dev/tmp/wt-fix-ec7`)

Filed by the lead from fixer `ec7`'s four messages (2026-11-10). 16 commits
on top of refs main 88bf0bc, nothing pushed, tree clean. No public function
removed or renamed. Independent verification: started (helper `v-ec7`).

Addendum (part 4), commits a3045f8, fd64ebf, 636fb8f and a CHANGELOG addition:
ec7-09 fixed: all 17 `table_a_*` lookups require their selectors (no default
row or set; `model_factor` of Table A.11 stays optional), first paragraphs
give the printed page and "Dimensionless; a recommended value: the National
annex may give another"; `design_approach_sets` has no default approach.
ec7-13: the result's note now opens with the caution (numbers unchanged; phi
30, gamma 120 pcf, H 20 ft: c 400 psf gives 61.1 deg, c 800 psf gives 46.9
deg). ec7-20: `heave_factor_of_safety` returns "fs": null instead of Infinity
when Q <= 0; three error texts. Final tests: test_eurocode_7_1.py 157,
test_eurocode_7_2.py 164, test_california_trenching_tables.py 102,
test_fema_p2082_tables.py 88, test_references_agent.py 36, audits pass.
App side: any `table_a_*` or `design_approach_sets` call without its selectors
is a TypeError; no file under funhouse_agent names them.

| ID | Status | What changed | Commit |
|---|---|---|---|
| ec7-01 | fixed | Eqs. C.1, C.2 with "- u" (Kp 3.0, u 49.05: 349.05 -> 201.90 kPa); Kpc docstring | 77334e0 |
| ec7-02 | fixed / raises | Table A.11 a step by the printed ">=" columns (n 4 min 1.40 -> 1.50); Table A.10 raises for n not printed, giving the neighbours | 2f58085 |
| ec7-03 | fixed | anchor block FS 1.5 -> 2.0 | dc4cf87 |
| ec7-04 | fixed (refuses) | Annex D: `shape` default None; "strip" without L, a misspelt shape and rectangular without L raise; result names the shape | 0eef6b6 |
| ec7-05 | fixed (raises below the first column) | Tables D.3, F.3; above the last column still clamped | 727abb8 |
| ec7-06 | fixed | Table 3-1 Very Loose row | dc4cf87 |
| ec7-07 | fixed (raises) | G.1, G.3, Eq. 4.5 outside printed ranges; `i_dmt` required | 4dfcf26 |
| ec7-08, -09 | left | library-wide result shape (lead) | |
| ec7-10 | fixed | `design_approach_sets("DA1")` returns both combinations | 3aaeb78 |
| ec7-11 | fixed | aliases "cohesion", "shaft", "concrete" removed (raise naming the choices) | 3aaeb78, ca8a37b |
| ec7-12 | left | bool parameters taking strings (dispatch, app side) | |
| ec7-13 | label only | `max_allowable_slope_angle` docstring caveat; code unchanged (owner) | 8bd8e92 |
| ec7-14 | fixed (no number changes) | printed notes and footnotes reach the results of D.1, H.1, D.5, min test count | 05f5db9 |
| ec7-15 | fixed | `seismic_design_category` requires S1; site class adds the classes at vs/1.3 and 1.3 vs | 86ca819 |
| ec7-16 | fixed | `at_rest_k0` refuses a slope with OCR > 1; `aep_cohesive_max_ordinate` `factor` required, Ka required for 4 < Ns < 6 | ca8a37b |
| ec7-17 to -20 | part | docstrings, printed pages, Table G.1 labels | 77334e0, 8bd8e92, ca8a37b, 0eef6b6, 4dfcf26 |
| ec7-21 | left | package descriptions; three printed rules not coded | |
| CHANGELOG | "Unreleased" entry | | b611bd4 |

Chapter text corrected in the same commits: eurocode_7_1 chapter09.json
(9.5.3), chapter13.json, chapter15.json (15-1); california_trenching
chapter10.json (10-2).

Tests (one file at a time, exit 0): test_eurocode_7_1.py 153,
test_eurocode_7_2.py 163, test_california_trenching_tables.py 98,
test_fema_p2082_tables.py 88, test_references_agent.py 36; audit_chapter_text
passes for all four packages. No app test run.

Calls that now raise or return a different shape (app side):
`equation_4_5_cu_from_dmt` (`i_dmt` required); `seismic_design_category`
(`s1` required); `aep_cohesive_max_ordinate` (`factor` no default; Ka for
4 < Ns < 6); `undrained_bearing_resistance`, `drained_bearing_resistance`
(`shape` default None, result has "shape"); `design_approach_sets("DA1")`
returns {"description", "combinations", "note"};
`table_a_10_correlation_ground_test` raises for n not printed; removed
aliases "cohesion", "shaft", "concrete"; new result keys (`notes`, `e_basis`,
`alpha_s_notes`, `table`, `printed_page`, `variability_criterion`,
`count_unit`, `footnote`, `estimated_vs_site_classes`, `other_combination`,
`note`, `pile_anchor_note`). App mention found: only
`funhouse_agent/tests/test_method_aliases.py:22` (not affected).

Not settled from the print: Caltrans Eq. 4-5-2 (owner); Table A.10 for n not
printed (refused; stepping down to the lower n is the alternative); Table F.3
between printed rows (no interpolation rule printed; still interpolates,
docstring says so); above the last printed column of D.3 and F.3 (clamped).

Noticed, not fixed: Table P.1 prints "0,95 << r <= 0,98"; the
`eurocode_7_1/tables.py` module docstring still describes the old method
("surviving OCR fragments"); `table_d4_pile_shaft_resistance` clamps above
15 MPa and at negative qc; `seismic_design_category` defaults
`risk_category` to "II" (echoed); Caltrans Eq. 8-3-4, Eq. 10-3-4 and the hard
layer within 0.7B not coded; Burland and Burbridge corrections docstring-only.
