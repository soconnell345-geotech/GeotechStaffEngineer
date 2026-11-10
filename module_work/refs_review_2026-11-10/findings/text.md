# text: the stored chapter text against the print — findings register

Parts 1 and 2 filed by the lead from reviewer `text`'s messages (2026-11-10);
parts 3 to 5 to be appended by the clerk. Full tables are the reviewer's
script output in `C:/dev/tmp/refs_review_2026-11-10/text/`:
`SAMPLE_TABLE.md` (all 574 checked sections, every error quoted),
`gen_tables.md` (Tables 3a-3d and Table 2: the 153 sections with a
substantive error; Table 3d lists every wrong DM 7 equation as stored /
printed / page), `scan4.json`, `results/`, `results2/`. Pages are PDF page /
printed page. None of the findings is on BACKLOG.

## Short answer

The references rewritten from the print in October (REF-25, 29, 45, 47, 52)
hold: 5 of 80 sampled sections have a substantive error, none a wrong number
or equation. The 19 references never checked do not: 77 of 301 sampled
sections (26 %). Worst exposure: DM 7.1 and DM 7.2, whose prose is the
print's own but whose equations were retyped from the PDF's garbled text
layer: all 344 stored equation entries checked on the rendered page, 38
sections hold formulas that are not the printed ones (about 40 numbered
equations and 10 table formulas), in `body` as well as `key_points` and
`equations`. GEC-11 and the FHWA pavements manual need rebuilding from the
print. GEC-10 chapters 6 and 8, which REF-52 skipped, are written from memory.

## Coverage

321 chapter files, 7,668 sections, 31 packages. Checked against the print:
574 sections (7.5 %): a seeded stratified sample (seed 20261110) of 301
sections in the 19 never-checked references and 80 in the 11 rewritten ones;
then every remaining DM 7.1 / DM 7.2 section that lists an equation (67 +
81), every remaining section of GEC-10 ch 2, 6, 8 (20), a second GEC-11
sample (16), the rest of the FHWA pavements manual (9; all 23 checked).
Method: one packet per section (stored JSON, PDF text, 130 dpi renders); 38
Opus checkers compared field by field, equations and table values from the
render; the reviewer re-read the print for 16 A findings, all as reported.
Mechanical scan of all 7,668 sections (part 5). NOT reached:
fhwa_soils_foundations (script-built from the print; scan clean); GEC-12 ch
5-18 and App. D; GEC-10 ch 10 and App. A; DM 7.2 App. B; GEC-9 App. A;
reference lists and glossaries of the UFC practice manuals; AASHTO 1993 by
eye only (12 sections). `qc_progress.json` lists only dm7_1/dm7_2 and every
chapter is `pending` (the weekend QC never ran, OWN-33).

## By package (substantive errors / sections checked; risk left)

| Package (sections) | How built | This review | Risk |
|---|---|---|---|
| fhwa_soils_foundations (511) | script, print's wording (REF-29) | not sampled; scan clean | low |
| gec_4, gec_5, gec_8, gec_9 | rewritten from the print (REF-52) | 0/8, 0/8, 1/8, 1/8 | low |
| gec_13, eurocode_7_2 | rebuilt from the print | 0/8, 0/8 | low |
| gec_12 | REF-45, REF-04/47, REF-52 | 1/6 (a remark, no wrong value) | low |
| gec_6, gec_7, micropile | REF-45 / REF-04 / REF-46 | 0/6, 0/5, 0/5 | low |
| gec_10 (395) | ch 2, 6, 8 NOT rewritten by REF-52 | ch 2 1/8; **ch 6 9/12 from memory; ch 8 2/3**; others 0/7 | high for ch 6, 8 |
| dm7_1 (457), dm7_2 (446) | bulk build: body from the PDF text layer; equations, summary, key points model-written; 72 container sections have no body | stage 1: 6/30, 10/30; every equation section checked | **HIGH for equations**; medium key points; low body prose |
| gec_11 (183) | bulk build, condensed paraphrase with content from memory | **16/40, 19 A errors** | HIGH: rebuild |
| fhwa_pavements (23) | bulk build, 23 sections for 132 printed headings | **10/23**; 9+ stored section numbers are another subject in print | HIGH: rebuild |
| california_trenching (51) | bulk build | 7/16, no A | medium |
| eurocode_7_1 (208) | bulk build, paraphrase | 4/24, 2 A | medium |
| ufc_pavement (317) | rebuilt 2026-07 by a model | 5/24, 2 A | medium |
| aashto_1993 (44) | model reading the scan | 3/12, 1 A | medium |
| fema_p2082 (28) | bulk build | 3/12, 2 A | medium |
| gec_14 (7) | bulk build | 4/7, no A | medium, little used |
| ufc_structural, ufc_collapse, gsa_collapse, em_2104, em_2107, wood_handbook | structural bulk builds | 3, 3, 1, 0, 4, 6 of 12 each | medium: ufc_structural, em_2107, wood; low: rest |
| ufc_concrete_practice, ufc_flexible_practice, ufc_stabilization | bulk builds, body near the print's wording | 1, 0, 1 of 12 each | low |

No chapter text: ufc_backfill, ufc_expansive (lookups only).

## Findings

- **text-01** (A) DM 7.1 and DM 7.2: 38 sections store equations that are not the printed ones (16 of 71 DM 7.1 equation sections, 22 of 88 DM 7.2); in 31 of the 72 error rows the wrong form is in `body` too. Full list: Table 3d in `gen_tables.md`. Named in the reviewer's message, DM 7.1: Eq. 1-4, 1-5 (`chapter01.json:589-605`; Is(50) = F P/d^2 with F = sqrt(De/50), stored P/(F d^2) with F = De/50); Table 4-2 (5 of 9 formulas); Eq. 4-8; 6-13 and 8-41 (1.99 x 10^2 stored as 10^-2); 8-8; 8-11 (31 + KD/(0.236 + 0.066 KD) stored as a sum: 45.6 against 61.2 degrees at KD 100); 8-13; 8-16; 8-26; 8-36 (inverted); 8-38; 8-39; 8-42, 8-43, 8-44. DM 7.2: Eq. 2-10; 3-2; 3-4; 4-25; Table 4-7 cofferdam checks (6 formulas); 5-5, 5-6, 5-11, 5-12; 5-21 (stored + ic where the print has - ic); 5-26; 5-34; 5-39; 6-24; 6-31; 6-41; 6-46, 6-47; 6-48 (swapped); 6-49, 6-50; 6-53; 6-60, 6-64 (square roots lost); 6-76; 6-79 (tcap = Pui/12 + 30 stored as Pui/30 + 12); 6-80. For most the Python lookup computes the printed form, so the helper can get the right number from the function and the wrong formula from the text for the same equation; where the code is wrong too it is in the dm71/dm72 registers. Fix: point fixes (retype every Table 3d formula from the rendered page in `equations[].description`, `key_points` and `body`, with a test pinning each to a worked value), not a rebuild. The fix branches dm71 and dm72 have corrected only the equations in their own findings.
- **text-02** (B; two cases A) 21 equation entries filed under sections that do not print them (dm7_1 10, dm7_2 11): `dm7_2/text/chapter06.json:276,285` "Eq. 6-1 ... E = 1820 sqrt(f'c)" (the printed Eq. 6-1 is the scour critical velocity; 1820 sqrt(f'c) is an unnumbered cell of Table 6-10); Eq. 6-80 listed under 6-8.3, printed in 6-7.4 whose stored body stops mid-sentence; Eq. 7-16; eight entries described "(force-injected from source text)" with no formula; two `implemented_in` names point at another equation's function.
- **text-03** (A) key points that misread a printed table: DM 7.1 Table 8-21 Begemann (`chapter08.json:857`; print E/Pa = 39.5 + C(N - 6) for N >= 15); Table 8-20 (`:837-838`; "Driven Piles" is a Poulos sub-column, values are E/Pa ratios, not pascals); "Table 4-7: Suggested Reading", "Table 4-8: Notation" (the printed Tables 4-7, 4-8 are tunnelling tables); DM 7.2 Table 5-22 "seven" problem soils (print: eleven); `chapter04.json:1195` "80 % pressure reduction" (Figure 4-54 note 3: moments are 80 % of simple-span moments); `chapter07.json:581` "2n" evaluations (print 2^n).

- **text-04** (A) GEC-11: equations, values and table numbers the print does not carry; REBUILD. `gec_11/text/chapter02..11.json`; 16 of 40 sections checked, 19 A errors, in body, key points and equations alike. `chapter03.json:95-108` (3.5.1): stored "Eq. 3-10: Tn = 0.86 x b x Ec x Fy", "3-11a: Tn = Ac x Fy", "3-11b: Tn = Ac x Fu"; print (Vol I 114/3-22): Ec = En - ER (3-10); Tal = Fy Ac/b (3-11a); Ptal = Fy Ac (3-11b); no 0.86, no Fu; wrong table numbers for resistance factors and corrosion rates; "phi_t = 0.75" (print 0.65 for grids on a rigid facing); an unprinted zinc-loss sum. `chapter03.json:60` (3.4): "alpha default 1.0 is conservative" (print 108/3-16: 0.8 geogrids, 0.6 geotextiles); Eq. 3-2 with an Rc term the print lacks. `chapter09.json:123,134` (9.2.6): "TS = (FSR - FSU) MD/(FSR D)"; print (Vol II 69/9-7): TS = (FSR - FSU) MD/D (stored tension two thirds of printed at FSR 1.5: unsafe); "0.9R for inextensible sheet" (print: R for both). `chapter09.json:206` (9.2.7) squeezing expression not Eq. 9-15 (80/9-18). `chapter07.json:260-303`: Eqs. 7-16, 7-17, 7-18 without the printed 0.8 or with terms on the wrong side (326-327). `chapter05.json:160,162`: "O95 <= D85" (print 252/5-26: AOS <= B D85, B 0.5 to 2). Values not in the printed section: 2.7 differential settlement 1/500 and 1/50; 4.5 "RF = 3.5"; 4.1.1 a deadline and a CDR requirement; "ASTM A572 Grade 65". Point fixes would leave the other 143 sections at the same rate.
- **text-05** (A) GEC-10 chapters 6 and 8 are written from memory (`gec_10/text/chapter06.json` 9 of 12 sections; `chapter08.json` 2 of 3). 6.3: "Maximum bar spacing should not exceed 6 inches" (print 149/6-5: clear distance not less than the greater of five times the maximum aggregate size and 5.0 in.: the stored limit points the other way). 6.4: spiral pitch limits and bar sizes not printed (152/6-8). 6.5: "minimum lap >= 2 ft", couplers "125 %": not printed (153/6-9). 6.8: hoops and cover misdescribed (161/6-17; cover 3, 4 and 6 in. by diameter). 8.2: "Minimum 5-10 load tests required": the printed 8.2 (225/8-17) gives no number. 8.1: flow chart blocks restated wrongly. Chapter 2 follows the page (one omission). Fix: rewrite chapters 6 and 8 from the print.
- **text-06** (A) UFC 3-301-01 Eqs. 16-18a, 16-18b wrong in the text (`ufc_structural/text/chapter02.json:122,131,132`) and the code (struct-16). Also B: 2-4 lists Figure 2-1 printed elsewhere; 7-1 widens a printed condition.
- **text-07** (A) UFC 3-250-01 Eq. 20-4 wrong in the text (`ufc_pavement/text/chapter20.json:246`) and the code (pave-04). Other ufc_pavement rows (B): `chapter09.json:102` "Apply Table 7-1 minimums" (minimum thickness is Table 7-2) and the 4 % cement limit dropped; 5-2 summary reverses the body's "cohesive fills at less than 90 percent" (print 33/13); 13-7 and 17-2 carry statements not on the page.
- **text-08** (A) FEMA P-2082 Section 19.3.2: two SSI damping equations wrong (`fema_p2082/text/chapter19.json:27,28,33` and the `latex` of 19.3-1, 19.3-3). Print (117-118/79-80): beta_0 = beta_f + beta/(T~/T)^2_eff <= 0.20; beta_f = [((T~/T)^2 - 1)/(T~/T)^2] beta_s + beta_rd. Stored: no 0.20 cap (unsafe); the factor on the wrong term; the tilde lost. Also B: chapter20 calls Site Class BC "the reference site condition" and omits 20.2.2 to 20.2.5; 21.2 stops before the deterministic lower limit; two ids are tables, not headings; 11.5 quotes Ie values not printed there.
- **text-09** (A) EN 1997-1: `chapter07.json:162-178` "Eq. 7.12" and "Eq. 7.16" are not the printed expressions (85/83; 87/85); `chapter14.json:28,29,38` Annex B.2 gamma_F misdefined (print 140/138: gamma_F = gamma_S;d x gamma_f); `chapter15.json:43-60` says C.3 to C.8 are "NOT LEGIBLE" and omits them, but they are legible (153-154/151-152); 6.5.2.2 body contradicts its own key points; 12.1 "should" stored as "shall".

- **text-10** (A) FHWA pavements manual: stored section numbers are other subjects in the print; REBUILD by script from the print (the fhwa_soils_foundations route). All 23 sections checked, 10 substantive. `chapter07.json:58` stored 7.4 "Stabilization ... and Geosynthetics" (printed 7.4 is "Compaction", 393/7-41); `chapter04.json:39` stored 4.7.3 "Soil Suitability" (printed "Subsurface Profile"); stored 5.3.3, 5.4.7 and five more likewise. `chapter04.json:59-70` (4.5.5): "CBR = 292/DCP^1.12 (ASTM D6951)" with id "Table 5-34 (DCP)"; the printed 4.5.5 (135-137) has no equation. `chapter07.json:37` (7.5.6): "clays and all silts F3"; Table 7-12 (412/7-60): all silts are F4 (silts under-treated). 51 listed objects outside their stored section.
- **text-11** (A) EM 1110-2-2107 paragraph 1.11 stored as empty (`em_2107/text/chapter01.json:130`); the printed 1.11 (11/2) carries requirements (non-carbon HSS, AISC Design Guide 27, Aluminum Design Manual). Three more sections drop printed limits (4.2, 4.3, 5.1). 4 of 12.
- **text-12** (A) Wood Handbook key points widen or change printed equations (6 of 12): `chapter08.json:16` Eq. 8-3 "for ALL dowel-type fasteners" (print: nails; bolts use 8-17a, 8-18a); `chapter09.json:239` Eqs. 9-39 to 9-41 "the same Euler form" (printed 9-41 is F''b1 = 1.44 E d2/(le d1)); B items in chapter 8 and chapter 5.
- **text-13** (A) UFC 3-250-11 section 4-2.1 (`ufc_stabilization/text/chapter04.json:125`): deep stabilization "NOT routine for airfield projects": the print never mentions airfields; the printed frost caution on lime is left out. Other 11 sampled sections clean.
- **text-14** (B) Caltrans: 7 of 16, no A (a "default fallback" slope, "Su = qu/2", "NDS, 2018", "AREMA Manual, Chapter 9", a wrong equation citation, appendix ids "12-3", "13-1" for A-3, C-1). No wrong factor or pressure found.
- **text-15** (B; one A) AASHTO 1993: 3 of 12. `chapter08.json` 3.2.2 (A): a flexible-design caution (modulus > 40,000 psi, printed in 3.1.5, 127/II-37) filed in the rigid design section. The rigid design equation and its worked example match the scan.
- **text-16** (B) single items: gec_14 ("must" for "should"; 4 of 7); `ufc_collapse/text/chapter04.json:78` ASCE 41 "Table 6-14", "6-15" (print 78/62: Tables 10-14, 10-15); 2-2.4.3 the ELR factors put on the strength where the print puts them on the demand; gsa_collapse a reason and a "Commentary C2.3" not printed; ufc_concrete_practice one added sentence.
- **text-17** (B) "should" stored as "must", and conditions dropped, in 14 sampled sections, mostly in key points and summary while the body keeps it (EN 1997-1 12.1; gec_14 2.1, 2.2; DM 7.1 4-5.3.1, 6-5.2.3; DM 7.2 4-6.2, 5-6.2.2; gec_11 2.7, 9.2.4; em_2104 3-8; em_2107 4.2; fhwa_pavements 7.5.6; gec_8 8.3.2.3; gec_9 11.2.1).
- **text-18** (B) objects listed in sections that neither print nor cite them: scan candidates 78 equations, 66 figures, 74 tables (fhwa_pavements 51, dm7_2 36, gec_11 25, dm7_1 21, ufc_pavement 18, gec_10 17); the checked and rebuilt references have none.
- **text-19** (C) section ids the print does not carry (appendix sections under a chapter number; table rows given section numbers; range ids; ordinals).
- **text-20** (C) 109 sections with no body (dm7_1 42, dm7_2 30, ufc_flexible_practice 23, others) carrying model-written summary and key points; 103 more with a body under 200 characters. Lists in `scan4.json`.

## Patterns

1. Equations retyped from a garbled text layer: 93 "wrong equation" records (dm7_2 41, dm7_1 26, gec_11 9, eurocode_7_1 5, fema 3, ufc_structural 3, ufc_pavement 2, wood 2). Every entry checked for DM 7.1 / 7.2; NOT done for gec_11 (94 entries, 31 seen), eurocode_7_1 (106, about 40 seen), ufc_pavement (52, 14), california_trenching (57, 19), wood_handbook (65, 24), ufc_collapse (63), fema_p2082 (22, 12), aashto_1993 (32, 9).
2. Content from memory in condensed references: 35 to 55 % of sections where the stored text is a short paraphrase of a long section (gec_11, fhwa_pavements, california_trenching, em_2107, gec_14, gec_10 ch 6 and 8); under 10 % outside equations where the body is close to the print's wording (dm7, the UFC practice manuals, em_2104).
3. The model-written fields (key_points, summary, applicability) drift from the body and carry most non-equation errors; `reference_search` shows the summary first.
4. Objects filed by proximity, not by where they are printed.
5. A section is silently partial: 44 "truncated" records; nothing in the returned section says it is a condensation.

## Estimates and ranking (part 5)

Counts: 20 findings (12 A, 6 B, 2 C), resting on 116 A and 241 B error records, each with stored text, printed text and PDF page (`SAMPLE_TABLE.md`). Totals: never-checked references 77 of 301 sampled sections (26 %); rewritten or checked 5 of 80 (6 %). Stage 2: dm7_1 equation sections 67 checked, 20 substantive; dm7_2 81, 31; gec_10 ch 2/6/8 rest 20, 10; gec_11 second sample 16, 6; fhwa_pavements rest 9, 4. Per-reference rates in the reviewer's Table 3a (`gen_tables.md`): dm7_2 33 %, dm7_1 20 %, california_trenching 44 %, gec_11 42 %, aashto_1993 25 %, eurocode_7_1 17 %, ufc_pavement 21 %, fema_p2082 25 %, fhwa_pavements 43 %, wood_handbook 50 %, em_2107 33 %, ufc_structural 25 %, ufc_collapse 25 %, ufc_stabilization 8 %, gec_14 57 % (7 sections), gsa_collapse 8 %, ufc_concrete_practice 8 %, em_2104 0 %, ufc_flexible_practice 0 %.

Where the errors sit: in the condensed references the body is as wrong as the model-written fields; in the near-verbatim ones (dm7, UFC practice manuals) non-equation errors sit in key points and summary, and equation errors are in every field including body.

Ranking by exposure (error rate x how often the reference is used) and route:
1. DM 7.2 and DM 7.1: highest use; one stored equation in eight wrong. POINT FIXES (Table 3d, text-02, text-03).
2. GEC-11: 40 %, every field. REBUILD from the print.
3. GEC-10 chapters 6 and 8: 11 of 15 sections. REWRITE the two chapters.
4. Caltrans Trenching and Shoring: 44 %, added statements, no wrong pressure found; heavily cited by the excavation-support module. Point fixes now, rebuild candidate.
5. EN 1997-1, UFC 3-250-01, AASHTO 1993, FEMA P-2082: 17-25 %; the A errors are equations. Point fixes plus an every-equation pass.
6. FHWA pavements manual: 43 %, section numbers unreliable, low use. REBUILD by script.
7. ufc_structural, ufc_collapse, em_2107, wood_handbook: 25-50 %, little used by the geotech agent. Point fixes for the A items.
8. Low: em_2104, gsa_collapse, the three UFC practice manuals, every reference rewritten or checked from the print. gec_14 (7 sections): rebuild is small.

Mechanical scan (all sections; AASHTO excluded): at chapter level the listed equation, figure and table numbers do exist in the chapter's pages (the "nowhere" cases are text-layer artifacts: GEC-8's equation numbers are images, the EN documents are OCR layers); at section level, candidates listed in a section that neither prints nor cites them: equations 78 of 1,918, figures 66 of 3,129, tables 74 of 1,353 (`scan4.json` "a2").

Right, though it looked wrong: DM 7.1 prints the labels of Eqs. 5-19, 5-28 as "(5-5-19)", "(5-5-28)"; GEC-8 and EN 1997-2 sampled sections clean on the page; GEC-11's lettered section ids are printed; UFC 3-250-01 5-2 "cohesive fills at less than 90 percent" is as printed; DM 7.1 Table 8-20 prints "Eu/Pa" under a drained-modulus title; AASHTO 1993's rigid design equation and example match the scan; the DM 7 lookups for Eqs. 8-11, 8-16, 8-36, 1-4, 2-10, 3-2, 5-21, 5-34, 6-24, 6-60, 6-64, 6-80 compute the printed form (only the stored text is wrong there).

Not done: equations outside DM 7 were only sampled; gec_11, eurocode_7_1, ufc_pavement, california_trenching, wood_handbook, fema_p2082, aashto_1993 each need the same every-equation pass.
