# struct: structural packages — findings register

Filed by the lead from reviewer `struct`'s four report messages (2026-11-10).
Packages: `em_2104`, `em_2107`, `ufc_structural`, `ufc_collapse`,
`gsa_collapse`, `wood_handbook` (383 functions). "pdf N" is 1-based,
"printed N" the number on the page. Evidence:
`C:/dev/tmp/refs_review_2026-11-10/struct/` (scan, fuzz, pagecheck, probes,
`png/`). Counts: **A 3, B 6, C 6.**

Coverage: scan of all 383 functions (no duplicate names, registry counts
equal the functions, no bare except); fuzz of 347 (key lookups raise with the
valid list almost everywhere); about 240 page citations checked; about 1,000
printed cells or expressions in 31 tables and 17 equations: **13 cells
wrong, 1 cell treated wrongly, 4 descriptions wrong.** All 92 rows of UFC
3-301-01 Table 3-1 match; 20 of 20 Wood Handbook checks match. Not reached:
em_2107 tables other than 4.1 and Eq 10.x; ufc_structural Tables C-1 to C-3,
E-1, 7-1, B-1; ufc_collapse chapters 6-8; GSA Table 12 onward; most
Wood chapter 9 stress functions; 36 functions with no test baseline.

## A

- **struct-01** `ufc_collapse/reinforced_concrete.py:162` `table_4_1_beam_other_modeling` (data `_TABLE_4_1_OTHER` ~:66). UFC 4-023-03 Table 4-1, pdf 79 / printed 63, acceptance (primary/secondary): shear s<=d/2 print 0.002/0.01, code 0.003/0.02; shear s>d/2 print 0.002/0.005, code 0.003/0.01; development s<=d/2 0.002/0.01 vs 0.003/0.02; development s>d/2 0.002/0.005 vs 0.003/0.01; embedment 0.01/0.02 vs 0.015/0.03. Ten cells 1.5-2x too permissive (the code returns a and b as acceptance). Module docstring admits the columns were never checked.
- **struct-02** RC beam flexure tables need V/(bw d sqrt(f'c)) in psi units; not said. `ufc_collapse/reinforced_concrete.py:114, :212`; `gsa_collapse/reinforced_concrete.py:116, :223`; aliases `beam_flexure_mfactor`. An SI ratio clamps to the "<=3" row (most favourable): vw 0.4 -> m 13.2; same beam in psi (4.8) -> 9.96 (33 % unconservative). Clamping not reported.
- **struct-03** `em_2107/tainter_gate_loads.py:546-558` (`TABLE_10_LOAD_COMBINATIONS`), print pdf 155-157 / printed 146-148: Eq 10.7 is "Gate Operated by Two Hoists" (code: gate jammed); 10.8 "Gate Operating on One Hoist" (code: gate jammed, unusual hydrostatic); "Gate Jammed" is Eq 10.9 (code: maximum operating equipment force, unusual); 10.9 and 10.11 are extreme per 10.3.5.2 / 10.3.6.2.2 (code: unusual). All 13 formulas match.

## B

- **struct-04** Steel connection m-factor and rotation formulas have no bound: `ufc_collapse/structural_steel.py:88, :145, :220, :281`; `gsa_collapse/structural_steel.py:138, :195, :270, :331`. d in mm gives m -58.9; d in cm gives GSA WUF 0.62; d > 40 in gives m < 1. Coefficients match the print (UFC pdf 85-86 / 69-70; GSA pdf 58-59).
- **struct-05** `em_2104`: 167 of 230 adapter parameters have no description; code assumes ksi, in, kip-in (`flexure_axial.py:51, :55`). `aci_beta1(28)` -> 0.65 (28 MPa should give 0.85). `em_2107`: 42 of 128 undescribed.
- **struct-06** `pdf_page` is 0-based in all six packages (about 240 citations); the app's convention is 1-based (GM-154, RI-01). (Library-wide: lead.)
- **struct-07** `em_2107/loads.py:114, :207`: Table 4.1 (pdf 36 / printed 27) T principal is printed 1.2 (footnote 4, one merged cell across usual/unusual/extreme); code has None ("NA").
- **struct-08** `ufc_structural/seismic_force_resisting_systems.py:38, :360`: Table 3-1 height limits are bare numbers in feet (footnote d, pdf 79 / printed 57: "For metric units, use 30 m for 100 ft and 50 m for 160 ft").
- **struct-09** `gsa_collapse/reinforced_concrete.py:165, :198, :325-333`: GSA Tables 6 and 8 (pdf 52, 54) print one acceptance column; code carries UFC's two. Table 8 cells: no continuity Vg/Vo<=0.2 print 0.020, code 0.015; development print 0.02, code 0.01; embedment print 0.03, code 0.02 (all conservative, cited as GSA's).

## C

- struct-10 wrong printed-page cites (em_2104 Table 3-2 is printed 24-25 = pdf 30-31; Tables 3-3/3-4 printed 26; UFC 4-023-03 Eq 3-6 printed 24; Wood Table 5-3a starts 5-4).
- struct-11 `em_2107/connections.py:23` `bolt_grade_check` returns permitted false for any unknown grade.
- struct-12 METHOD_INFO types string parameters as "float" (137 registry entries across the six).
- struct-13 `ufc_collapse/tie_forces.py:337, :382` default lp = 3.3 ft (conservative; documented).
- struct-14 `em_2107/loads.py:271` `principal_load_factor_self_straining` ignores `minimum`.
- struct-15 results with no unit names (pattern P2).

## Patterns

P1 0-based pdf_page everywhere. P2 numeric results with no unit-named key (registry entries): em_2104 26, em_2107 31, ufc_structural 20, ufc_collapse 43, gsa 35, wood 36; only wood names units in keys as a rule. P3 parameters with no description: em_2104 167/230, em_2107 42/128, ufc_structural 16/80, ufc_collapse 13/153, gsa 16/118, wood 24/275. P4 unbounded linear formulas (8 functions). P5 silent clamping to the printed grid in 8 RC flexure functions. P6 US-unit inputs with no unit in the description. P7 only the wood functions tell quantity, unit, source and limits.

## Functions where an SI caller gets a plausible wrong number

1. `vw_ratio` tables (struct-02). 2. em_2104 (struct-05): `aci_beta1`, `kd_table_d1`, `design_singly_reinforced`, `design_doubly_reinforced`, `balanced_eccentricity_*`, `compression_controlled_capacity_*`, `compression_steel_stress_at_ku`, `balanced_strain_kb`. 3. `table_3_1_seismic_system` heights (struct-08). 4. peripheral tie forces lp default (struct-13). 5. steel connection m-factors with d in cm (struct-04).

## Right, though it looked wrong

Key lookups strict; number-keyed tables refuse out-of-range input; RC clamping matches the printed <= and >= rows (only the units problem remains); Wood table_5_13's 25 % for unlisted species is printed and flagged; UFC Table 4-2 m = 16/19 as printed; GSA Table 7 differs from UFC's and each follows its own print; all steel coefficients match.

## Added by the lead after the register (2026-11-10)

- **struct-16** (A; found by reviewer `text`, reproduced and print-read by the lead) `ufc_structural/risk_category_and_loads.py:245, :~265` `wind_speed_conversion_asd`, `wind_speed_conversion_fastest_mile`: code computes (0.6*V)**0.5; UFC 3-301-01 (2023, Change 4) Eq. 16-18a/b, PDF p. 41 (printed 19), as typeset: Vasd = sqrt(0.6) V; Vfm = (sqrt(0.6) V - 10.5)/1.05. At 150 mph: 9.49 and -0.96 mph against 116.2 and 100.7. The stored chapter text carries the same wrong formula. The reviewer missed it: its print check was of table cells; the formula functions of all six structural packages are being re-read against rendered pages in the fix stage.
