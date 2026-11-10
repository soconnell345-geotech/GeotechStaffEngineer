# dm71 fix table (branch `fix/review-dm71`, worktree `C:/dev/tmp/wt-fix-dm71`)

Filed by the lead from fixer `dm71`'s five messages and addendum (2026-11-10).
18 commits on 88bf0bc, HEAD 327a531, nothing pushed, tree clean. Independent
verification: to be started by the clerk.

Addendum (commits acc3ad8, 2a907d9, bcdedaf, 327a531): Eq. 8-44 takes the
saturation in percent (w a decimal; S x 100 inside; refuses S above 100 % or
not above 0; example 7.860e-11 m/s, decimal reading 1.4224e-07; before the
review 1.52e-08). Eq. 3-2, 4-7, 4-8 docstrings reworded (coded form kept; "the
owner is to revisit"; 4-7 and 4-8 not called a misprint). Range refusals added
for dm71-25, each naming range and page: Hazen D10 outside 0.1 to 3 mm
(stated, PDF 486); Eq. 8-13 and 8-16 PI above 100 (Figure 8-13); Eq. 8-2, 8-3
N60 above 50 (Figure 8-2); Eq. 8-5, 8-6 N1,60 above 60 (Figure 8-4); Eq. 8-4
N60 above 60 or sigma'v above 2.90 atm (Figure 8-3, the chart the equation
approximates); Eq. 5-9 t below 0.1 year. Added without a ruling: Eq. 8-43
refuses a saturation above 1.0 or not above 0 (LEAD: allow up to 1.10 in both
8-43 and 8-44, since a computed saturation a little over 100 % occurs from
measurement scatter; to be applied by an Opus successor after the check).
Nothing printed settles Eq. 8-44's unit: no figure, table or worked example in
the manual uses Eq. 8-43 or 8-44 with numbers. Tests: dm7_1 chapters 1-8: 47,
29, 49, 167, 269, 121, 61, 301; the others as before; audit passes. No app
caller of anything changed in the four commits.

Judgment calls on this branch (for the owner's later item): Eq. 3-2; Eq. 4-7;
Eq. 4-8; Eq. 8-44 (percent); Eq. 8-43 (w decimal, k in m/s as printed);
Kozeny-Carman fractions must total 1 or 100; Figure 5-6 mu1 band middle and
linear interpolation in L/B; plotted-range refusals taken from an axis or
curve end (only Hazen's and Eq. 8-38's ranges are stated in words).

Earlier text of this file (before the addendum):
14 commits at the time of the five-part table. Every
equation was read from a 165 dpi crop of the rendered page, every table from
a rendered crop, the two charts traced on the page's embedded image.
"PDF p. (printed p.)" in `ufc_3_220_10_2026.pdf`.

All 17 A findings fixed or rebuilt (Eq. 8-44 set to refuse by the fixer; the
lead has since ruled it takes the saturation in percent). B: dm71-19, -21,
-24 fixed; -18 and -20 code left with docstrings corrected; -22, -23 (cut),
-25 the lead's. No public function removed or renamed.

| ID | Status | Function | Print | Before -> after |
|---|---|---|---|---|
| dm71-01 | fixed | `cpt_friction_angle_robertson_campanella` | Eq. 8-8, 430 (402) | at qc/sigma'v 100: 9.25 -> 40.51 deg |
| dm71-02 | fixed | `fully_softened_friction_angle` | Eq. 8-13, 435 (407), also printed on Fig 8-13 | PI 40: 45.57 -> 26.83 deg |
| dm71-03 | rebuilt | `figure_8_46_f` | Fig 8-46, 472 (444): one curve PI 15 (6.75) to 64.5 (4.24) | PI 20: 400 -> 5.51; refuses outside 15 to 64.5 |
| dm71-04 | rebuilt | `table_8_6_stark_hussain` | Table 8-6, 445 (417): 16 rows | CF > 50 %, 400 kPa, LL 80: 26.5 -> 10.3 deg; keys changed |
| dm71-05 | fixed | `spt_undrained_strength_stroud_butler` | Eq. 8-26, 456 (428) | N 20, PI 30: 4.26 -> 93.8 kPa (the app's own `spt_to_cu` already had the printed form) |
| dm71-06 | fixed | `hydraulic_conductivity_carrier_beckman` | Eq. 8-42, 490 (462) | 2.5e-4 -> 3.5e-9 m/s; non-positive bracket refused |
| dm71-07 | fixed | both Kozeny-Carman functions | Eq. 6-13, 328 (300); Eq. 8-41, 490 (462) | 7.2e-8 and 7.7 cm/s -> 0.0716 cm/s from both; fractions must total 1 or 100 |
| dm71-08 | fixed | `cbr_from_spt` | Eq. 8-39, 485 (457); Fig 8-57 | N 10: 2.5e9 -> 6.14; N >= 300 refused |
| dm71-09 | fixed / ruled | `hydraulic_conductivity_benson_landfill` (Eq. 8-43: w as a decimal), `..._benson_trast` (Eq. 8-44) | 491 (463); the page prints k in m/s for both and no unit for w (lead saw the page) | 8-43: OverflowError -> 3.6e-8 m/s; 8-44: percent reading per the lead |
| dm71-10 | fixed | `well_resistance_factor` | Eq. 5-26, 296 (268) | z = Lm: 0 -> 1.257 |
| dm71-11 | rebuilt | `figure_5_6_mu0`, `figure_5_6_mu1` | Fig 5-6, 258 (230), nu = 0.5 only | mu0 at D/B 20: 0.62 -> 0.859; mu1 square at H/B 5: 0.82 -> 0.645; refuses nu != 0.5 and outside the axes; below H/B 1.5 the curves run in one band (+/-0.03) |
| dm71-12 | rebuilt | `table_8_9_shansep_m` | Table 8-9, 453 (425): seven rows by source and soil | parameter `soil` (was `test_type`), returns a dict |
| dm71-13 | rebuilt | `table_8_24_cbr_dcp` | Table 8-24, 484 (456): 13 rows | six unprinted rows removed |
| dm71-14 | rebuilt | `table_4_3_trench_fill` | Table 4-3, 223-224 (195-196) | saturated clay K mu' 0.029 -> 0.11 |
| dm71-15 | rebuilt | `table_4_9_tunnel_behavior` | Table 4-9, 233 (205) | printed rows; required `soil_type` |
| dm71-16 | fixed | `cbr_from_dcp_nazzal` | Eq. 8-38, 484 (456): valid 6.3 to 67 mm/blow | DCP 20: 9.03 -> 11.65; refuses outside |
| dm71-17 | fixed | `cpt_friction_angle_mayne` | Eq. 8-7, 428 (400) | 32.98 -> 36.29 deg |
| dm71-18 | code left | Eq. 3-2, 4-7, 4-8 | 169 (141); 223 (195); 225 (197) | docstrings and chapter text state the printed forms |
| dm71-19 | fixed | `drain_spacing_factor(approximate=False)` | Eq. 5-23, 295 (267) | n 5: 0.906 -> 0.936 |
| dm71-21 | fixed | Table 4-2 angle descriptions | 214 (186) | descriptions only |
| dm71-24 | fixed | `table_5_6_Calpha_Cc` | Table 5-6, 282 (254) | shale row, +/-0.01, printed source |

Chapter text corrected (chapter03, 04, 05, 06, 08.json), including Eq. 8-16
(text wrong, code right) and three `implemented_in` links.

Tests (one file at a time, exit 0): dm7_1 chapters 1-8: 47, 29, 48, 165, 268,
121, 61, 288; cover_citations 29 + 1 xfailed; references_agent 36; five
retrieval files. No app test run.

Changed keys, arguments or returns: `table_8_9_shansep_m`,
`table_4_9_tunnel_behavior`, `table_8_6_stark_hussain`, `table_8_24_cbr_dcp`,
`table_4_3_trench_fill`, `figure_5_6_mu1` (nu default 0.5),
`hydraulic_conductivity_benson_trast`, the Kozeny-Carman pair,
`cbr_from_dcp_nazzal`, `cbr_from_spt`, `figure_8_46_f`, `figure_5_6_mu0`.

App files that name them: `calc_package/tests/test_calc_package.py:493-505`
(`plot_figure_8_46`, signature unchanged);
`funhouse_agent/tests/test_top_level_params.py:340-341` (parameter names
unchanged); prose in CONSOLIDATION_PLAN.md and CONSOLIDATION_CHANGES.md;
`webapp/tests/page_fingerprints.json` may want re-recording when the
submodule pointer moves; the library's CLAUDE.md test counts.

Not settled from the print: Eq. 8-44's unit of w (ruled: percent); Eq. 8-43
gives about 3.6e-8 m/s at ordinary liner inputs, 30 times the usual liner
criterion (source may be in cm/s; kept as printed); Eq. 4-7, 4-8 (kept as
coded); Eq. 5-23 follows the typeset form (the page's sentence about the
approximation's error then holds from n 4.6 and 12.5, not 4 and 12); Figure
5-6(a) below H/B 1.5 and between curves (linear in L/B kept).

Noticed, not fixed: the dm7_1 chapter text's equations were never checked
wholesale (worth a backlog item); `audit_chapter_text.py` checks that an
`implemented_in` target exists, not that it is the right one;
`hydraulic_conductivity_kozeny_carman` defaults S to 6.0 (no printed default);
`pytest.approx` with `rel` alone does not test a value near 1e-9 (default
absolute tolerance 1e-12), so other tests of very small values may pass for
that reason; `well_resistance_factor`'s Lm description is not the page's.
