# gec: eleven FHWA lookup packages — findings register

Filed by the lead from reviewer `gec`'s ten report messages (2026-11-10).
Packages: gec_4, gec_6, gec_7, gec_8, gec_9, gec_10, gec_11, gec_12, gec_13,
micropile, fhwa_pavements. Code paths relative to
`geotech-references/geotech_references/`; PDFs in `geotech-references/docs/`.
"p." printed page, "PDF" 1-based viewer page. Evidence:
`C:/dev/tmp/refs_review_2026-11-10/gec/` (notes.md, out_*.txt, img/).
Counts: **A 8, B 29, C 13.** None on BACKLOG.

Coverage: 144 of 144 functions read and run; 119 print-checked here; 25
verified earlier (REF-42/49/57/60/62/63), 7 of those spot-checked, all hold.
Equations were read from the text layer and cross-checked against the print's
worked example or chart; NOT yet rendered as typeset (to be done in the fix
step): GEC-10 Eqs 10-16, 10-17, 10-18, 10-21, 10-22; GEC-8 side resistance
pp. 83-84; pavements Eqs 5.9, 5.11, 5.12, 5.13, 5.15 and Table 5-34.

Lead's own check against the print: gec-03 (page rendered), gec-04, gec-05,
gec-07, gec-08, gec-18 (calls reproduced, print text read).

## A

- **gec-01** `gec_6/figures.py:98-166` `figure_5_19_hough_bearing_capacity_index`. GEC 6.pdf Fig 5-19, p.91 (PDF 97). Not the printed curves; curve names 3-6 not the print's (print, bottom to top: Inorganic SILT; Sandy CLAY; Well graded fine to medium silty SAND; Clean well graded fine to coarse SAND; Well graded silty SAND & GRAVEL; Clean uniform medium SAND). `coarse_sand` holds the top curve: 32-38 % too much C' at N' 21-41. Below N' 15 the code is 30-60 % low. `n_prime=32, well_graded_sand_gravel` -> 96.6; the manual's example (p. B-28, PDF 186) reads 110. "sand" -> sandy clay curve by first substring. Bare float.
- **gec-02** `gec_7/figures.py:20-76` `figure_4_3_friction_angle_spt`. GEC 7.pdf Fig 4.3, p.79 (PDF 119). Not the printed curves: 3 to 8 degrees low for N60 >= 12; 2 to 4 degrees HIGH for N60 < 10 at stress ratio >= 2 (unsafe). `n60=22, ratio 2.0` -> 33.6; print 37.0. Runs to N60 70 (axis ends at 60). Bare float.
- **gec-03** `micropile/figures.py:26-57, :63-65, :100-107` `figure_5_23_limiting_lateral_modulus`. Micropile Manual Fig 5-23, p.5-70 (PDF 162); rule p.5-68 (PDF 160). 9.625x0.47 casing coded 159 psi, print 71; hollow bars 73/53 and 103/51 swapped (print 165 and 350); 7"x0.73" casing (159) and #20 bar (610) missing; docstring states the buckling rule reversed; "5.5", "8", "10", "20" match the wrong section by substring.
- **gec-04** `gec_12/tables.py:135-166` `table_7_2_resistance_factor_field`. GEC 12 vol 1.pdf Table 7-2, p.226 (PDF 272). Values match the page; "static load test" and "dynamic testing" return 0.80 (the combined row; alone 0.75 and 0.65, or 0.75 on 100 % of piles); "signal matching" returns the tension row. Bare float.
- **gec-05** `gec_12/tables.py:309-346` `table_7_16_soil_setup_factor`. Table 7-16, p.299 (PDF 345). "silty clay" and "sandy clay" return the Clay row (2.0; print Silt - Clay 1.0, Sand - Clay 1.5); "silty sand" returns Silt (1.5; print Sand - Silt 1.2). Footnote "Confirmation with Local Experience Recommended" and site counts dropped.
- **gec-06** (as printed; print looks wrong; OWNER) `gec_12/tables.py:214-228` `table_7_10_brown_method_factors`. Table 7-10, p.260 (PDF 306); Eq. 7-21 p.258 (PDF 304): fs = Fvs (Ab + Bb N60), ksf, N60 3 to 50. Compression rows print Bb = Ab for Gravelly Sand to Boulders (0.888) and Rock (2.89); tension rows print 0.0. At N60 = 30: 27.5 and 89.6 ksf; Table 7-8 (p.253, PDF 299) limits unit shaft resistance to at most 2.4 ksf. Second oddity: the three Vibratory rows are typeset under "Tension"; the code files them under compression (fix: file under tension).
- **gec-07** `gec_13/tables.py:625-707` `table_3_1_lightweight_fill`. GEC 13 Vol 1.pdf Table 3-1, p.3-1 (PDF 151). Print: Geofoam 0.70-3.00; Cellular Concrete 20-80; Wood Fiber 35-55; Tire Shreds 37-73; ESCS 37-65; Fly Ash 70-90; Boiler Slag 60-90; Expanded Air-Cooled Slag 69-94 (pcf), with a specific gravity column. Code: 4 of 8 rows differ (Foamed Concrete 24-55, Blast Furnace Slag 60-75, ESCS 40-65, TDA 37-55); notes not the table's; specific gravity dropped; "cellular concrete" -> [].
- **gec-08** (as printed; print looks wrong; OWNER) `fhwa_pavements/equations.py:162-228` `cbr_from_dcp`, `resilient_modulus_from_dcp`. FHWA-NHI-05-037 Table 5-34, p.5-53 (PDF 231): "CBR = 292/(DCP^1.12)", "DCP = Penetration index, in./blow", ASTM D6951. 0.5 in./blow -> CBR 634.7, Mr 158,867 psi. Table 5-28 (p.5-39, PDF 217) typical field CBR 3 to 80. Chapter 4 (p.4-39, PDF 141) gives the relation after Webster et al. (1994) with no unit. 12.7 (mm) -> 16.9.

## B

- **gec-09** `gec_13/tables.py:200-312` `table_1_3_by_function`: rows 5-8 not Table 1-3's (pp.1-15, 1-16; PDF 29-30); two printed rows missing ("Decrease imposed loads", "Transfer embankment loads to more competent layers").
- **gec-10** `gec_6/figures.py:23-88` `figure_4_1_relative_density_spt` (p.34, PDF 40): wrong for Dr >= 80 % (N 30 at 1.0 ksf -> 87.5; the print's 100 % curve passes there); held at 15 and 100 unsaid; chart covers 0-6 ksf, code 0.5-4. Conservative.
- **gec-11** `gec_7/figures.py:86-137` `figure_5_11_basal_heave_nc` (p.124, PDF 164): not the printed curves; default be_le = 0 up to 17 % low (conservative).
- **gec-12** `gec_12/figures.py:703` `figure_7_18_adhesion_factor`: default stratigraphy `sand_over_stiff_clay` is the most favourable (0.754 vs 0.363 for stiff clay only).
- **gec-13** `gec_6/figures.py:174-221` `figure_4_2_friction_angle_from_density` (p.35, PDF 41): answers unit weight/Dr combinations outside the printed band; Dr 0 line up to +2.4 degrees.
- **gec-14** `gec_10/figures.py:67-131` `su_uu_to_ciuc`, `su_uc_to_ciuc` (Eqs 10-16, 10-17, p.10-16, PDF 296): no range check; can return a negative strength (10, 700 -> -1030).
- **gec-15** `gec_10/tables.py:190-194` `table_8_4_resistance_factor` (p.8-14, PDF 222): one-word key answered from the first row ("side" -> cohesionless 0.55; "uplift" -> 0.45).
- **gec-16** `gec_6/tables.py:582-627` `table_5_15_friction_factor` (p.99, PDF 105): "sand" -> top row 0.55-0.60; "fine sand" -> 0.45-0.55 (print's fine sand row 0.35-0.45). Unsafe for sliding.
- **gec-17** `gec_7/tables.py:213-250` `table_4_3b_elastic_modulus_spt` (p.82, PDF 122): "sand" -> silts multiplier (7,661 kPa; clean sand 13,406). Bare float.
- **gec-18** `micropile/tables.py:653-680` `table_5_8_epsilon_50_stiff` (p.5-56, PDF 148): interpolates between bins the print gives as ranges (su 75 -> 0.006; print 0.007); no 200-300 row in the print.
- **gec-19** `micropile/tables.py:733-777` `table_5_10_soil_modulus_k_clay`: returns a cyclic k for Soft and Medium, which the print leaves blank.
- **gec-20** `gec_11/tables.py:725-760` `table_4_4_traffic_surcharge` (p.4-17, PDF 159): the abutment (perpendicular to traffic) table returned for any wall; walls parallel to traffic are heq = 2.0 ft (p.4-16). Held at the 5 ft row below 5 ft.
- **gec-21** `gec_11/tables.py:49-61` `table_2_1_min_reinforcement_length` (p.2-43, PDF 83): two invented seismic rows (print: one row "0.8 to 1.1 H"); 8 ft minimum on every row not in the table.
- **gec-22** `gec_7/tables.py:16-27` `table_4_2a_spt_soil_properties` (p.78, PDF 118): invents phi_min 25 and phi_max 50 for "< 30" and "> 45"; drops note (3) capping sands at 40 and gravelly soils at 45.
- **gec-23** `gec_10/tables.py:548-596` `aashto_reliability_index`: not printed in GEC-10; linear interpolation of pf (27-37 % high). RULING: remove.
- **gec-24** `gec_10/tables.py:496-540` `table_11_2_group_efficiency_cohesionless`: values are Section 11.4.1.2's (p.11-6, PDF 340); there is no Table 11-2. RULING: cite the section; keep the name.
- **gec-25** `gec_9/tables.py:214, :239-245`: Table 7-1 beyond 5B holds the 5B row unsaid; Tables A-1, A-2 answer outside 0.5-4 ton/ft2 from the end bin.
- **gec-26** `fhwa_pavements/tables.py:524, :570` `table_5_49_drainage_modifier_mi` (p.5-87, PDF 265): as FHWA prints, differs from the library's AASHTO 1993 Table 2.4 in two cells (Poor, Very Poor, < 1 %).
- **gec-27** `fhwa_pavements/equations.py:498-533` `modulus_subgrade_reaction_from_cbr`: relation (k = Mr/19.4) not printed in FHWA-NHI-05-037. RULING: remove.
- **gec-28** `micropile/tables.py:233-308` `table_4_2_rebar_properties` (p.4-27, PDF 83): offers bars the table lacks (#14 Grade 550), computed yields differ from printed (#18 1342.1 vs 1335), 2.5 in Grade 550 bar missing.
- **gec-29** (OWNER) `gec_7/tables.py:691` `table_4_9_site_coefficient_fpga`: GEC-7 Table 4.9 (p.94, PDF 134) heads the columns PGA <= 0.25 ... >= 1.25; GEC-12 Table 7-27 (p.380, PDF 426) prints the same numbers under PGA < 0.1 ... > 0.5. Site class D at PGA 0.30: 1.56 vs 1.2.
- **gec-30** `gec_4/tables.py:69-70` Table 6: "very stiff" answered from the Hard row.
- **gec-31** `gec_6/tables.py:253-256` Table 5-7: "rock"/"hard" -> strongest row.
- **gec-32** `gec_6/tables.py:297-300` Table 5-9: quality label one class low between rows.
- **gec-33** `gec_6/tables.py:384` Table 5-12: linear interpolation across decades of L/W (9-11 % low). Left for the result-shape pass.
- **gec-34** `gec_11/tables.py:893-922` Table 4-7: key `combined_static_phi` is a copy of static, beside the print's combined static/earthquake 1.00.
- **gec-35** (pattern) 27 lookups return a bare number, several in tsf, ksf, percent. Lead.
- **gec-36** `gec_12/tables.py:40` Table 7-1: null for the printed text cells ("Differs in Manual (1)", "not in AASHTO"). Left for the result-shape pass.
- **gec-37** (pattern) 21 filter lookups return [] for an unknown filter (including the print's own row names).

## Rulings given (fix stage)

gec-06, gec-08: keep as printed, add a `print_note` to the returned dict and
the docstring's first paragraph stating only what the page shows; no refusal;
vibratory rows filed under tension. gec-23, gec-27: remove (rule 2).
gec-24: as proposed. gec-29: keep as printed, state the conflict; owner.
gec-33, gec-36: result-shape pass (lead).

## C (part 8)

- gec-38 `gec_4/tables.py:247` Table 7 rows carry `recommended_FS` 3.0; the table prints no factor of safety (p.74 gives 2.0 for intermediate geomaterials).
- gec-39 `gec_4/tables.py:247, :560`: "limestone" -> soft limestone, "shale" -> soft shales; Table 20 words not the print's ("watertight weld", "Corrugated HDPE or polypropylene"; source "modified after PTI, 1996").
- gec-40 `gec_6/tables.py:414, :477` Tables 5-13, 5-14: one-letter keys match ("s" -> diabase); "weathered granite" -> intact granite.
- gec-41 `gec_7/tables.py:850` Table 5.1 (p.108, PDF 148): short_term_excavation 1.25 (print "1.25-1.33 (3)"); basal_heave_short_term fs_seismic 2.3 carries note (5), long-term only.
- gec-42 `gec_8/equations.py:19` (pp.83-84, PDF 106-107): drops "N60 not depth corrected" and "cohesionless materials only". Read from code.
- gec-43 gec_10: `table_10_2_nc_base_clay` takes psf while siblings take kPa; `equation_10_22_caving_rock_side` holds RQD < 20 at the 20 row; `equation_10_21_rock_socket_side` no cap at concrete strength (p.10-21); `figure_10_6_alpha_clay` answers su/pa < 0.25 (alpha 1.448).
- gec-44 gec_11: Table 4-6 two-decimal computed values where the print has one decimal; Table 3-9 Type 1/2 defined only in the docstring; Table 3-1 notes and Fig 4-10 footnote dropped.
- gec-45 gec_12: Fig 7-17 holds D/b at 10 and 40 unsaid; Fig 7-9 "pile" -> pipe curve; Table 7-8 interpolates between delta rows given by density class; Table 7-11 "silt" -> 2.5 (the row "Sandy silt and silt" is 1.5).
- gec-46 `gec_13/tables.py:17-164` Table 1-2: two rows cut short.
- gec-47 micropile: Table 4-5 fy 248 vs note (4) 241 MPa; Table 5-12 sand rows follow the ksf column (the print's kPa column is garbled; no note); Table 5-11 words not the print's.
- gec-48 fhwa_pavements: `cbr_from_plasticity_gradation` accepts p200 = 0.35 as a percent; `resilient_modulus_from_cbr` no upper check; Eq. 5.11 cited for Eq. 5.12; `pdf_page` 0-based.
- gec-49 `describe_method` reports a text key as float for gec_4's six unannotated functions (app side; core-17).
- gec-50 gec13 alias `ddc_depth` wanted by two functions (same function; harmless).

## Patterns (part 9; every function listed in the reviewer's message and in `gec/notes.md`)

1. Bare number as the whole result: 27 (gec_6 5, gec_7 5, gec_10 3, gec_12 10, micropile 4). Models in the library that do it right: gec_13 `equation_4_1_ddc_depth`, every fhwa_pavements function.
2. Empty list for an unknown filter: 21 (gec_10 1, gec_11 11, gec_12 1, gec_13 8). Model: gec_6 `table_5_10_fill_bearing_capacity` raises with the valid list.
3. Partial key answered from the first matching row: 19 (gec_4 2, gec_6 5, gec_7 6, gec_10 1, gec_12 4, micropile 1). Model: micropile `table_5_3_alpha_bond` raises "Ambiguous soil_type ... Matches: ...".
4. Values or words the cited print does not have: 13.
5. Outside the printed range: 14 (7 held at the end value unsaid, 7 computed with no check).
Also: defaults that pick the more favourable case (gec_12 `figure_7_18_adhesion_factor`; micropile `table_5_9_soil_modulus_k_sand`, default above the water table; micropile `table_5_3_alpha_bond`, default Type B).

## Aliases and adapter output (part 10)

165 aliases over eleven adapters; one collision (harmless). Six aliases mean a different table in different adapters (`p_multiplier`, `bearing_capacity_factors`, `spt_soil_properties`, `elastic_modulus_spt`, `lateral_resistance_factor`, `group_efficiency_cohesionless`). Aliases that hide the subject: gec10 `resistance_factor`, `by_category`; gec13 `by_function`, `applicability`, `unit_cost`; gec11 `traffic_surcharge`; gec12 `adhesion` beside `adhesion_factor`, `correction_factor`, `kd`, `nq`, `alpha_t`; gec6 `friction_factor`; micropile `elastic_modulus`, `group_efficiency`, `fixity`, `classification`. Twelve lookups through the adapter: six give a number and nothing else.

## Right, though it looked wrong

fhwa_pavements Table 5-49 is as FHWA prints; `cbr_from_plasticity_gradation` takes P200 as a decimal where the page says "% passing" (with the percent the equation gives CBR under 1 for any plastic soil; the result's equation string says so; RULING: add a `print_note`); Eq. 5.12 same numbers; Table 5-34's other formulas as coded (read as typeset). Micropile Table 5-3 all 28 bond ranges; Fig 5-23's other 15 bars; Eq. 5-29 reproduces the printed bars. GEC-12 Table 7-1 rows and Eq. 7-21 as typeset; `figure_7_10_to_13_kd` agrees with Table 7-6. GEC-10 su conversions reproduce the printed example; Table 10-2 in psf as printed; Table 8-4 rock side 0.50 (text says 0.55; settled by GM-278). GEC-7 Table 4.9 matches its page. GEC-13 Tables 1-6, 2-1, 4-3, 8-1, 8-6 and Eq. 4-1. GEC-8 and GEC-9 tables. GEC-4 Tables 6, 7, 8 values.
