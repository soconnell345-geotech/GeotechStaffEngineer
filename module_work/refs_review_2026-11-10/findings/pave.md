# pave: UFC pavement family, backfill, expansive soils, AASHTO 1993 — findings register

Filed by the lead from reviewer `pave`'s report messages (2026-11-10).
Packages: ufc_pavement (53 functions), aashto_1993 (52), ufc_backfill (8),
ufc_expansive (9), ufc_stabilization (17), ufc_flexible_practice (36),
ufc_concrete_practice (18). Code root `geotech-references/geotech_references/`;
PDFs in `geotech-references/docs/`; pages 1-based (the code's own `pdf_page`
in ufc_pavement, ufc_stabilization, ufc_flexible_practice,
ufc_concrete_practice and aashto_1993 is 0-based). Evidence:
`C:/dev/tmp/refs_review_2026-11-10/pave/` (probe*.py, probe*_out.txt,
helper_tail.txt). Counts: **A 8, B 12, C 13.** None on BACKLOG (related:
REF-58, GM-298, GM-303, REF-12, REF-31, OWN-115, OWN-21).

Coverage: backfill 8/8 and expansive 9/9 against the held PDFs; ufc_pavement
45 of 53 (8 verified by history); aashto_1993 38 of 52 (14 by history; of the
38, 30 against a rendered page, 8 by code read or worked example only:
minimum_layer_thicknesses, effective_modulus_subgrade_reaction,
frost_heave_rate_group, frost_susceptibility_classification,
swell_probability_guidance, swell_rate_constant_theta,
total_environmental_loss, weighted_swell_parameters). An Opus sub-helper
checked stabilization 17, flexible 36, concrete 18 read-only (no code run);
5 of its findings re-checked against the PDF, all hold. Six history-verified
functions spot-checked against printed examples, all match.

Lead's own check against the rendered print: pave-02, pave-03 (PDF 127,
printed 107) and pave-04 (PDF 114, printed 94) confirmed.

Calc-module use: `pavement_design/ufc.py` uses none of the functions in the A
findings. aashto_1993 findings in functions the module calls: pave-10
(`drainage_mi_flexible`, `drainage_cd_rigid`; the module takes the midpoint, so
the number is unchanged, only its basis text shows "1.4-1.35"); pave-11's clamp
in `layer_coefficient_a1_asphalt` (`flexible.py:129`); pave-28
(`relative_damage_rigid`, `k_loss_of_support`); pave-31 (no floor or check in
`layer_coefficient_a2_granular_base`, `layer_coefficient_a3_granular_subbase`,
`structural_number`). pave-12's unit traps are on the helper path only.

## A

- **pave-01** `ufc_pavement/tables.py:854-941` `figure_15_1_cracking_projection_factor`. Fig 15-1, `ufc_3_250_01_2016.pdf` PDF 70 (printed 50). Curves for k >= 150 too low: at 1e5 passes k 300 print 0.83 / code 0.69; k 400 0.753 / 0.53; k 450 0.698 / 0.42; k 500 printed from about 8,500 passes, code raises below 1e6. Result claims "~+/-0.05 on F". `flexible_overlay_of_rigid_thickness(0.69, 14, 1, 8)` 4.98 in; with 0.83, 10.86 in.
- **pave-02** `ufc_pavement/equations.py:1150-1198` `drainage_layer_thickness_required`. Eq 20-17, PDF 127 (printed 107): H = 2 F R L t / (1.7 ne L + k i t). Code omits the 2 (0.1840 ft against 0.3681 ft).
- **pave-03** `ufc_pavement/equations.py:1201-1238` `drainage_layer_thickness_simplified`. Eq 20-18, same page: H = F R t / (0.85 ne). Code has 0.85 in the numerator (0.2833 ft against 0.3922 ft).
- **pave-04** `ufc_pavement/equations.py:809` `effective_porosity`. Eq 20-4, PDF 114 (printed 94): ne = 1 - [gamma_d/(Gs gamma_w)](1 + Gs We). Code computes gamma_d We/gamma_w (the water retained). Also found independently by reviewer `text`; the stored chapter text carries the same wrong formula; cited page (pdf 111, printed 92) is not the equation's.
  `effective_porosity(110, 2.65, 62.4, 0.02)` -> 0.0353; print 0.2995. ne feeds T50/T85 (time about 8x too short, unsafe) and H.
- **pave-05** `ufc_pavement/tables.py:470` `table_9_1_equivalency_factor`. Table 9-1, PDF 47 (printed 27), "Asphalt-stabilized": All-bituminous concrete 1.15 / 2.30; GW, GP, GM, GC 1.00 / 2.00; SW, SP, SM, SC "*" / 1.50 ("* Not used for base course material."). Code has one row ("asphalt", "all") 1.15 / 2.30: GW base -> 1.15 (print 1.00); SM subbase -> 2.3 (print 1.50); SM base -> 1.15 (print: not used). 12 in of conventional subbase on SM becomes 5.2 in instead of 8.0 in.
- **pave-06** `ufc_backfill/tables.py:12-60` `table_compaction_requirements`. Held print `ufc_3_220_04fa_2004.pdf` (TM 5-818-4) para 5, PDF 29 (printed 5-2): CE 55 (modified effort) at 90 % in open areas, 95-100 % under or near structures. Code: every row "ASTM D698" ("Standard Proctor"), 95 / 90 / 85 %. Adapter cites "UFC 3-220-04N", not held (REF-31); results carry no page. Confidence medium (04N cannot be checked; the values are in no held document).
- **pave-07** `ufc_expansive/tables.py:84-90` `table_grade_beam_void_space` (echoed by `table_foundation_selection`). `ufc_3_220_07.pdf` (TM 5-818-7) PDF 58: void space beneath grade beams "6 to 12 inches or 2 times the predicted total heave, whichever is larger". Code: 0 / 0 / 50 / 100 / 150 mm by swell class.
- **pave-08** [sub-helper, confirmed] `ufc_flexible_practice/tables.py:1223-1265` `table_6_2_cold_mix_asphalt_selection`. Table 6-2, `ufc_3_250_03_2018.pdf` PDF 117 (printed 103): kerosene L per metric ton (gal per ton): cold 8.3 (18.8¹) and 2.0 (4.5¹); moderate 7.1 (15.4¹) and 1.7 (3.7¹); hot 6.3 (12.5¹) and 1.5 (3.0¹); footnote 1: amount when the mixture is to be stockpiled. Code returns only the first of each pair and its docstring says those apply when stockpiled: 2.0-2.3x too little.

## B

- **pave-09** `ufc_pavement/tables.py:130-215` `table_5_1_compaction_depth`. Table 5-1, PDF 34 (printed 14): separate ESAL bins for flexible and rigid pavements (rigid from "<1,300"); code has the flexible bins only and no pavement-type input; bins use <= where the print uses "<".
- **pave-10** `aashto_1993/tables.py:490-596` `drainage_mi_flexible`, `drainage_cd_rigid`. Tables 2.4 and 2.5, `aashto1993.pdf` PDF 115-116 (II-25, II-26): values match; each range is printed high-low ("1.40-1.35") and stored as (min, max) in that order, so `mi_min` 1.4, `mi_max` 1.35. Calc module calls both (`flexible.py:196-199`, `rigid.py:64-71`) through `midpoint_range`, so its number is unchanged; its basis text prints "1.4-1.35".
- **pave-11** Silent clamps against REF-58's refuse-below-range rule: `figure_10_1_k_on_base` (`ufc_pavement/tables.py:694-760`; Fig 10-1, PDF 52: k 25-300, thickness 0-60 in; (10, 10) -> 65.0 as (25, 10); (400, 10) -> 350.0; (100, 80) -> 420); `layer_coefficient_a1_asphalt` (`aashto_1993/tables.py:284`; Fig 2.5, PDF 107: digitised from 100,000 psi; 50,000 and 3,000 psi both -> 0.2 with no note; CALLED by the calc module at `pavement_design/flexible.py:129`, so a refusal there needs an app-side message); `figure_15_1_cracking_projection_factor` (passes above 1e7 held unsaid); ufc_stabilization `table_3_5_emulsified_asphalt_requirements` (`tables.py:530-577`; printed rows 0-25 % passing No. 200; 28 % read as the 25 % row).
- **pave-12** SI values into US-unit inputs give believable wrong answers (22 functions; names in part 7): `flexible_sn_from_w18` with MR 50,000 (kPa) -> SN 2.612 against 5.288 for 7,250 psi; `max_serviceability_loss_frost("poor", 1.5)` (m) -> 0.6 against 1.968 for 4.92 ft; `table_9_1_maximum_joint_spacing(200)` (mm) -> 20 ft; `free_draining_layer_required(200, 550)` -> false (true in the intended units). Lead's result-shape pass.
- **pave-13** ufc_expansive: classification and most index-property functions are not from the print. `ufc_expansive/tables.py:14-80`; `equations.py:11, 65, 95, 146, 197`. Print `ufc_3_220_07.pdf` Table 4-1, PDF 31 (printed 4-1): Low (<0.5 % swell; LL <50, PI <25), Marginal (0.5-1.5 %; LL 50-60, PI 25-35), High (>1.5 %; LL >60, PI >35). Code: PI 30 -> "medium", "4 - 8" %; LL 55 -> "high". Not in the print: `swell_pressure_kPa` (Komornik-David; bare float 712.3 for (40, 16, 20); suspected about 16x high), `free_swell_percent` (Seed 1962; (45) -> 23.35 while the module docstring says about 6.1 %), `table_active_zone_depth` (para 5-4c has no climate table), `table_foundation_selection` (does not match Table 6-1, PDF 44), `pier_minimum_embedment_m`, `heave_prediction_mm`, `activity_index`'s "highly_active" class.
- **pave-14** ufc_backfill: values the held print (`ufc_3_220_04fa_2004.pdf`) does not have. `ufc_backfill/tables.py:62-160`; `equations.py:11-185`. `table_backfill_material_classification`: Types I-III not printed; code calls CH "unacceptable" where the print (PDF 23, printed 3-6) says CH "might be considered suitable". `table_maximum_lift_thickness`: print's Table 5-1 (PDF 30, printed 5-3) gives compacted lifts 12/8 in rubber-tired, 8/6 crawler, 6/4 power tamper, 6 sheepsfoot, no limit for vibratory; code differs. `table_compaction_equipment_selection`, `table_drainage_requirements`, `filter_criteria_check` (Cu <= 20), `compaction_induced_pressure_kPa` (sqrt(2 q gamma K0/pi)) not printed. Earlier decision REF-31 (done 2026-11-06) kept the package cited to "UFC 3-220-04N" with the gap recorded. The Whole Building Design Guide's search (lead, 2026-11-10) returns only "UFC 3-220-04FA Backfill for Subsurface Structures" (16 January 2004, INACTIVE) for this number. RULING: rebuild from the held 04FA and cite it; remove what it does not print.
- **pave-15** (OWNER; confidence low to medium) `ufc_stabilization/equations.py:146-182` `equation_cement_pi_limit_table_2_3`. `ufc_3_250_11_2020.pdf` PDF 21 (printed 13), Table 2-3 "Guide for Selecting a Stabilizing Additive", last footnote line as typeset: "(c) PI <= 20 + 50 - percent passing No. 200 (0.075 mm) sieve" (no parentheses, fraction bar or divisor on the 200-dpi render). Row 1C's portland cement restriction is printed only as "2. (b)", and footnote (b) is about classification, not a formula. Worked example (PDF 16, printed 8-9): SC soil, 25 % passing, PI 9, "less than the equation indicated in footnote b"; no number stated for the limit. Code follows the page literally (70 - P200: 60 at 10 %, 45 at 25 %); its docstring says "hand-verified" as 45, which the print does not state. Why it looks wrong: "20 + 50" written as two constants; two reviewers recall, from memory only, the source manual (TM 5-822-14 / AFJMAN 32-1019, not held) dividing (50 - P200) by 4, which would give 30 at 10 % passing (the printed form would then be the unsafe side). Fix stage: keep as coded, state the literal printed line; no formula from memory.
- **pave-16** [sub-helper, confirmed] `ufc_flexible_practice/tables.py:1223-1245` Table 6-2: the immediate-use/stockpile emulsion split is invented; print gives grades by climate (cold "MS-2h, SS-1h"; moderate "MS-2 to MS-2h, SS-1 to SS-1h"; hot "MS-2, SS-1") and footnote 2.
- **pave-17** [sub-helper, confirmed] `ufc_concrete_practice/tables.py:635-682` Table 9-1 "Maximum Joint Spacing" (`ufc_3_250_04_2024.pdf` PDF 122, printed 104): fields named `joint_spacing_min_ft`/`max_ft` for a range of the maximum; exactly 12 in goes to the ">12" row (the print's "9 to 12" row includes 12); footnote 1 (10 to 15 ft for roads and parking lots).
- **pave-18** [sub-helper] `ufc_concrete_practice/tables.py:91-130` Table 3-2 (PDF 33, printed 15): footnotes 2 (1.5 % for crusher dust) and 3 (slag exempt) dropped; stricter than the print.
- **pave-19** [sub-helper] ufc_stabilization `table_a1_equivalency_factors`: the "limit the cement content to 4 percent or less" note is in the docstring, not the result (Table A-1, PDF 71, printed 63).
- **pave-20** [sub-helper] `ufc_flexible_practice/tables.py:285, 336` Table 2-8: Section 1 air-void targets (4 HMA, 6 sand asphalt) left out.

## C (part 5)

- pave-21 `ufc_pavement/equations.py:934-976` `time_for_50pct_drainage`: Eq 20-11 (PDF 118, printed 98) T50 = ne D/(2 i k); code omits the 2 (8.993 against 4.496; conservative); the T85 note is wrong.
- pave-22 `figure_10_1_k_on_base` reads 10-20 % low (conservative): at 10 in, print / code k 300 385/350; 200 300/255; 100 200/160.
- pave-23 `ufc_pavement/tables.py:144-153, 223` Table 21-2 (PDF 144, printed 124): the cohesive 80 % column (9 ... 27 in) is missing; docstring says none is printed; bins use <= where the print uses "<".
- pave-24 `ufc_pavement/equations.py:723` `taylor_permeability`: Eq 20-3 (PDF 113, printed 93) prints gamma e^3/(mu (1 - e)); code uses e^3/(1 + e) (the physically correct Taylor form) without saying so.
- pave-25 `ufc_pavement/tables.py:949` Table 16-1 (PDF 74, printed 54): docstring says the 20 ft maximum applies at freezing index "< 1,800"; the print says "1,800 or more". Values match.
- pave-26 page cites: ufc_pavement Chapter 20 equations cited 1-4 pages early (Eq 20-1/2 cited pdf 108/printed 89, printed on PDF 112/92; Eq 20-4 cited 111/92, on 114/94; Eq 20-17/18 `pdf_page` 127 is the 1-based page under a 0-based convention); aashto `aggregate_loss_army_corps` cites pdf_page 102 (0-based) for text on PDF 102 (II-12); flexible fog-seal rate cited p.70 (PDF 85, printed 71); slurry-seal CSA p.124 (formula on p.123).
- pave-27 `aashto_1993/equations.py:803-890` `aggregate_loss_army_corps`, `annual_aggregate_loss_kenya` (`aashto1993.pdf` PDF 102, II-12): the print attributes GL = (B/25.4)/(0.0045 LADT + 3380.6/R + 0.467 G) to a Brazil study, not the US Army; the first printed equation GL = 0.12 + 0.1223 LT is not implemented; Kenya prints ".092T", code uses 0.92 T. Calc module: not used.
- pave-28 unprinted closed forms, both CALLED by the calc module: `relative_damage_rigid` (`composite_k.py:530`; Fig 3.5, II-41) u_r = (D^0.75 - 0.39 k^0.25)^3.42 / 100, disclosed in the result, up to 12.5 % off chart reads; `k_loss_of_support` (`composite_k.py:625`; Fig 3.6, PDF 132, II-42) k^slope(LS): LS = 1 matches the printed example (540 -> 170); (2000, 2) -> 99.3 and (2000, 3) -> 34.0 against the reviewer's quick chart read of about 115 and 38 (code low, conservative; read uncertain).
- pave-29 `funhouse_agent/adapters/_reference_common.py:36-39`: unannotated text parameters advertised as "float" (pavement 18 methods, aashto 15, backfill 5, expansive 3). App side (= core-17).
- pave-30 no result from ufc_backfill or ufc_expansive (17 functions) carries a reference or page; two return bare floats.
- pave-31 nonsense inputs unchecked, all three CALLED by the calc module (which passes psi and inches): `layer_coefficient_a2_granular_base(200)` -> -0.404, a3 likewise (no floor at low moduli); `structural_number` with mm -> 72.0.
- pave-32 [sub-helper, confirmed] `ufc_stabilization/tables.py:586-591` Table 3-6 (PDF 38, printed 30): High is LL > 60 and PI > 35; code uses >=; a negative input raises StopIteration.
- pave-33 [sub-helper, not re-checked] smaller items: concrete `table_3_5` Type IP given Type P's remark; `table_d1` "retarded concrete" under longitudinal (print: transverse); `edge_slump_tolerance` print conflict p.83 "6 mm (0.15 in.)" vs p.124 "6 mm (0.25 in.)" not recorded; `table_3_1` freezing index has no unit; printed "-" returned as null; stabilization "No. 4" raises in Tables 3-3, 3-4; Table A-1 base factor null with no note; flexible `table_2_3` PTI exactly 44.4 -> "hot" (print: warm), "**" footnote dropped; `table_6_3` drops targets 4 and 75; `slurry_seal_surface_area` takes a dict (JSON keys arrive as strings).

## Named lists (part 7)

Clamps that hold at the edge WITH a note and are fine: the esal_* lookups, `k_loss_of_support` (LS outside 0-3), `max_serviceability_loss_frost` (depth above 10 ft).

pave-12, the SI-unit traps: 24 functions (not 22). ufc_pavement (10): `figure_10_1_k_on_base`, `figure_15_1_cracking_projection_factor`, `figure_f1_rigid_thickness`, `free_draining_layer_required`, `rigid_overlay_fully_bonded`, `rigid_overlay_partially_bonded`, `rigid_overlay_non_bonded`, `flexible_overlay_of_rigid_thickness`, `insulation_mean_soil_temperature_estimate`, `insulation_initial_temperature_differential`. aashto_1993 (7): `flexible_sn_from_w18`, `rigid_d_from_w18`, `layer_coefficient_a1_asphalt`, `k_loss_of_support`, `k_rigid_foundation_correction`, `effective_roadbed_resilient_modulus`, `max_serviceability_loss_frost`. ufc_backfill (1): `compaction_induced_pressure_kPa`. ufc_expansive (2): `swell_pressure_kPa`, `pier_minimum_embedment_m`. ufc_flexible_practice (2): `table_2_3_asphalt_grade_by_pti`, `table_2_5_asphalt_grade_by_freezing_index`. ufc_concrete_practice (2): `table_9_1_maximum_joint_spacing`, `table_3_1_weather_severity`.

Patterns: P1 values the held print does not contain under a UFC citation: 14 functions (backfill 6, expansive 8), plus two that contradict the print. P2 Chapter 20 equations: 4 of 18 wrong (20-4, 20-11, 20-17, 20-18), and 20-3 departs silently. P5 misleading labels: 3 (mi, Cd, concrete joint spacing). P6 results with no reference: 17.

## Right, though it looked wrong

ufc_pavement kPa/mm = psi/in x 0.271 is correct; Table 19-2 S1 for GW at 3-6 %; AASHTO frost-heave constant 0.02 (PDF 405, G-11) and swelling constant 0.00335 (PDF 401, G-7); printed examples reproduce (stage_reliability 97.47 vs 97.5; flexible_sn_from_w18 4.976 vs 5.0; rigid_d_from_w18 9.719 vs the nomograph's 10.0, OWN-115; composite_k 390.8 vs 400; k_rigid 303 vs 300); AASHTO Table 2.7, k = MR/19.4 (PDF 134, II-44) and the Fig G.7 slopes match exactly; mi, Cd, J values match (only labels swapped); the stabilization PI limit is faithful to the page; slurry Eq B-1 0.099941 against the printed 0.99941 is a correction the result states; a misspelt keyword raises TypeError with a did-you-mean and a misspelt text value raises ValueError listing the valid values, in all 7 packages.
