# pavement_design against AASHTO 1993, UFC 3-250-01 and FHWA-NHI-05-037, as printed

BACKLOG GM-298 (takes in GM-35), 2026-10-31. Read from the PDFs in
`geotech-references/docs/`: the AASHTO Guide for Design of Pavement
Structures (1993; a scan, read as page images), UFC 3-250-01 (14 November
2016) and FHWA-NHI-05-037 (Geotechnical Aspects of Pavements). Page numbers
are the printed ones, with the PDF page where it helps.

Scope: everything in `pavement_design/` (traffic, flexible, rigid, composite
k, environmental losses, the UFC flexible and rigid methods, mixed traffic,
the comparison, calc steps, plots) and
`funhouse_agent/adapters/pavement_design_adapter.py`. Each equation, chart
read, default and limit the module uses was read against the print. Where a
chart was the thing used, it was read off the page image (pixel reads at the
scan's resolution) and checked against every worked example the document
prints for it. What was fixed and what was only noted is below.

## AASHTO 1993: layer coefficients for treated bases, Figures 2.8 and 2.9 (fixed)

- **Print.** Figure 2.8 (printed II-23) and Figure 2.9 (II-24) are parallel
  scales: a2 on the left, 7-day UCS (2.8) or Marshall stability (2.9)
  beside it, read straight across. The dashed line on Fig 2.8 reads a2 0.22
  at 800 psi.
- **Module before.** The reference layer's three-point tables, which put a2
  0.10 at 200 psi / 200 lb (the print gives about 0.126; 0.10 sits near 30
  psi / 120 lb) and held 0.22 above 800 psi (the print rises to 0.25 at
  1,000 psi), with a "sparse chart read" warning.
- **Now.** Read off the print at every labelled tick (UCS 200-1,000 psi ->
  a2 0.126-0.25; Marshall 200-1,800 lb -> 0.126-0.32), linear between ticks.
  Below the lowest tick is refused ("give a directly"); above the top tick
  holds the top value with a warning.

## AASHTO 1993: layers stiffer than 40,000 psi, Section 3.1.5 (fixed)

- **Print.** Section 3.1.5 (II-37): the Figure 3.2 procedure "should not be
  applied to determine the SN required above subbase or base materials
  having a modulus greater than 40,000 psi"; the layers above are set by
  "cost effectiveness and minimum practical thickness".
- **Module before.** Solved SN over any base given a modulus (a granular
  base of 45,000 psi set the AC from it) and refused a treated base with no
  modulus.
- **Now.** No SN is solved over a layer stiffer than 40,000 psi; a treated
  base with no modulus counts as one (every modulus on Figures 2.8/2.9 is
  higher). The layer above takes the larger of `min_thickness_in` and the
  Section 3.1.4 minimum (error if neither exists), and a note quotes II-37.

## AASHTO 1993: the layered check, Figure 3.2 note 2 (fixed)

- **Print.** Figure 3.2 (II-36): the SN provided above each layer must be
  at least the SN required over it (SN1* >= SN1, SN1* + SN2* >= SN2).
- **Module before.** Check mode compared the total SN against the roadbed
  requirement only, so a thin AC on a thick base read adequate.
- **Now.** Check mode tests the criterion over every layer whose modulus
  (40,000 psi or less) is given; `sn_stack[i]` carries `sn_provided_above`
  and `adequate`, and the overall `adequate` needs all of them.

## AASHTO 1993: defaults and their basis (wording fixed, values kept)

- So 0.45 flexible / 0.35 rigid: the guide's total standard deviation with
  traffic (Section 2.1.3 step 3, II-9/10); ranges 0.40-0.50 / 0.30-0.40
  (I-62). The notes said "midpoint of the range"; they now give the II-9/10
  basis.
- Drainage m = 1.0 and Cd = 1.0 when nothing is given: the value for the
  AASHO Road Test conditions (Section 2.4.1, II-25). The notes said Table
  2.4/2.5 "fair", which those tables do not say.
- Rounding: the guide rounds flexible layers "to the nearest 1/2 inch"
  (Section 3.1.4, II-34) and slabs "generally ... to the nearest inch, but
  ... slip form pavers may permit 1/2-inch increments" (Section 3.2.2,
  II-44). The module rounds up to 0.5 in (Figure 3.2 asks the value used to
  be at least the value required); the docstrings and adapter now quote
  both and give `thickness_increment_in=1.0` for whole inches. Owner
  question 1.
- Growth factor: Table D.20 (D-24); the reference was cited elsewhere.

## AASHTO 1993: composite k, Figure 3.3 (fixed; takes in GM-35)

- **Print.** Figure 3.3 (II-39), a two-quadrant chart: Dsb across, Esb
  curves above, MR curves below, a turning line and k_inf diagonals. Its
  own example Esb 20,000, MR 7,000, Dsb 6 -> 400 pci; Table 3.3 (II-43)
  rows 1,100 / 160 / 230 / 410.
- **Module before.** The reference layer's fit through the four Table 3.3
  rows, all at Dsb 6 in, scaled by a three-point guess for Dsb. Away from
  those rows it departs from the chart: at Dsb 6, Esb 400,000, MR 5,000 it
  gives 3,026 pci where the chart gives about 490; its Dsb scaling is too
  flat.
- **Now.** `pavement_design/composite_k.py` reads Figure 3.3 off the page
  (every curve at every even Dsb gridline, the chart's own construction) and
  runs the Table 3.2 worksheet with Figures 3.4-3.6 from the reference
  layer. Anchors: 391 / 1,085 / 154 / 230 / 391, within 5 % (the guide's
  own two reads of one point differ by 2.5 %). The Table 3.3 worksheet: avg
  u_r 0.613 (printed 0.60), k 522 (540, -3.3 %), LS-corrected 165 (170,
  -2.7 %); V-074.
- **Warning added.** Section 3.2.1 (II-37/44): for an unbound subbase the
  ratio Esb/MR should not exceed 4. The module warns when a season exceeds
  it. The guide's own Table 3.3 has 6.0 in March (Esb 15,000 on MR 2,500),
  so it warns on the guide's example. Owner question 2.
- Figure 3.4 (reference layer) is exact at Dsg 5 ft and not re-read.
  Figure 3.5's closed form matched a pixel read within about 2 % for D 6-14
  in, k 15-1,950; Figure 3.6's slopes within about 8 % for k >= 30.

## AASHTO 1993: Figure G.3, swelling VR (noted; GM-35)

- **Print.** Figure G.3 (Appendix G) example: PI 50, average conditions,
  5-ft layer -> VR 0.83 in; Table H.1 (Appendix H) agrees.
- **Reference layer.** Its Fig G.3 read gives VR 2.5 to 4 times the chart.
  The module does not call it (VR is the caller's input); DESIGN.md says
  so. REF candidate.

## AASHTO 1993: base and subbase moduli from CBR in the comparison (fixed)

- **Print.** Figure 2.6 (II-19, base) and Figure 2.7 (II-21, subbase) carry
  a CBR scale beside the modulus and a2/a3 scales; the Road Test pairs are
  base CBR 100 with 30,000 psi and subbase CBR 30 with 15,000 psi (II-20).
  FHWA-NHI-05-037 Table 5-34 (p. 5-53) gives Mr = 2555 CBR^0.64 (NCHRP
  1-37A) for subgrade soils.
- **Module before.** `compare_flexible_pavement_methods` put the FHWA
  correlation on the base and subbase too: base CBR 80 -> 42,205 psi, a2
  0.175, and over 40,000 psi (so, with the Section 3.1.5 fix, no SN over
  it), where Figure 2.6 reads about 0.13.
- **Now.** Base and subbase moduli are read across Figure 2.6 (base CBR
  20-100) and Figure 2.7 (subbase CBR about 9-100); off the scale is refused
  ("give the modulus directly"). Road Test checks: CBR 100 base -> 29,800
  psi, CBR 30 subbase -> 14,800. The FHWA correlation stays for the
  subgrade. The note states both values. Owner question 3.

## UFC 3-250-01: Figure E-1 (fixed)

- **Print.** Figure E-1 (printed 192): thickness of cover above a layer of
  each CBR for 10 to 10,000,000 passes of the 18-kip single axle. Appendix G
  reads it in G-1 (CBR 3, 1,000,000 -> 16.4 in; CBR 4, 1,395,400 -> 14 in)
  and G-3 (5,000,000: CBR 10 / 7 / 4 / 25 -> 7.8 / 9.8 / 14.5 / 3.4 in).
- **Module before.** The reference layer's `figure_e1_flexible_thickness`:
  exact at the Appendix G anchors, up to 1.7 in under the curves between
  them (CBR 10, 1,000,000: 5.7 in where the chart reads 7.4; CBR 2-3 at
  10,000,000, 1.6-2.2 in high).
- **Now.** `pavement_design/ufc_e1.py` follows each printed curve (CBR 1 to
  where it reaches zero), interpolating in log(passes) between curves.
  Every Appendix G read within 0.1 in. G-8.10 says CBR 3.5 (the frost
  support index) at 1,200,000 passes gives 16 in; the curves (and the
  reference layer) give 15.1: an inconsistency in the print, noted.

## UFC 3-250-01: the 4 in subbase minimum (fixed)

- **Print.** Para 7-5 (printed 20): "a minimum of 4 in of drainage layer
  and 4 in of subbase (separation) course for most pavements". G-3 ends
  3.5 in AC, 4 in base, 4 in subbase.
- **Module before.** A subbase came out at whatever the cover cascade left
  (G-3: 0.5 in).
- **Now.** A subbase is at least 4 in, with a note citing p. 20; G-3
  reproduced (3.5 / 4 / 4). Owner question 4. Table 7-2 (printed 21) cells
  all match the print; Eq 13-1 matches G-5.2 (hd 8.3 -> ho 6.6).

## UFC 3-250-01: Figure F-1 (fixed)

- **Print.** Figure F-1 (printed 224), worked as para 13-3 (printed 37)
  says: flexural strength across to the k curve, down to the pass-level
  line, across to the thickness scale. Appendix G reads it in G-7 (650 psi,
  k 100, 20,000,000 -> 8.1 in), G-8.11 (k 325, 1,200,000 -> 6.3) and G-8.12
  (k 50, 1,200,000 -> "a pavement slab 8 in thick"). Para 13-3: round up in
  0.5 in steps; "The minimum thickness of plain concrete is 6 in".
- **Module before.** The reference layer's `figure_f1_rigid_thickness`, a
  two-anchor model that barely moves with flexural strength (k 100,
  1,000,000: 6.6 in at 400 psi, 6.2 at 900, where the chart gives 9.9 and
  6.0) and gives 6.5 in at G-8.12's k 50. No 6 in minimum.
- **Now.** `pavement_design/ufc_f1.py` reads the chart: the eight
  pass-level lines are straight (fitted, residual under 1 px); the seven k
  curves were tracked row by row and fitted (residual 0.3 px). G-7 8.1,
  G-8.11 6.3, G-8.12 7.8 (8 in once rounded up). k above 500 is taken as
  500 (Table 10-1 note 2, printed 31); below 25 refused. The slab is rounded
  up to 0.5 in and raised to 6 in where less. Table 10-1 cells match the
  print. The UFC plots now draw the module's E-1 and F-1 reads.

## UFC 3-250-01: frost (one fix, rest noted)

- Table 19-3 (printed 80) cells match: F1/S1 9.0, F2/S2 6.5, F3/F4 3.5.
  The reference layer cites it as pdf_page 100, printed 81; it is on
  pdf_page 99, printed 80.
- **PFS (fixed).** Table 19-2 footnote *** (printed 70): PFS is "possibly
  frost-susceptible, but requires laboratory test to determine frost design
  soils classification". The module said "no frost adjustment required";
  it now warns that the laboratory test sets the group and computes no
  frost section.
- **Table 19-2 rules (noted).** The reference layer classifies USCS symbols
  the print does not list (SM under 6 %, SC under 15 %, GC 6-20 %, GM 3-6 %,
  OL, OH, PT, GW-GC, GP-GC) by its own extension, and always puts SM over
  15 % in F4 and CL in F3 (the print has both F3 and F4 rows for each, the
  F4 rows printed "very fine silty sands" and "Clays, PI > 12"). F3 and F4
  both give 3.5 in Table 19-3, so the flexible frost section does not move.
  REF candidate.

## AASHTO 1993: checked, matching

- Table D.4 (D-6), flexible single axles, pt 2.5: every cell of the
  reference layer's table matches the print.
- Table D.20 (D-24) growth factors: nine cells within 0.006 (V-075).
- Table D.21 (D-25) worksheet: per-row ESALs exact; the column sums to
  43,797,514 where the total line prints 43,772,314 (a 25,200 slip in the
  print; text D-22 says 43.8 million) (V-075).
- Appendix H flexible example (H-1 to H-7): traffic 18,598,900, effective
  MR 5,695, performance period 13.21 yr, SN 3.18 / 4.53 / 5.60; D 8 / 7 /
  12 in (the guide rounds 11.04 to 11; the module rounds up) (V-076).
- Table 3.1 (II-34) cited at the right page in calc_steps.
- Figure 2.5 a1 from EAC: the reference layer's 150,000 psi anchor reads a
  little off the chart (REF candidate, small; no output change in the
  saved examples).

## What moved (master -> this branch)

| Case | Before -> after |
|---|---|
| AC on cement-treated base UCS 600 on subbase | ValueError (needs modulus) -> AC 3.5 / base 14.5 / subbase 16.0 in |
| AC on bituminous-treated base | ValueError -> AC 3.5 / base 12.0 / subbase 15.5 in |
| Cement-treated base UCS 400 | a2 0.160 -> 0.155 |
| Granular base given Ebs 45,000 psi | AC 6.0 / base 9.0 -> AC 3.5 (II-37 minimum) / base 15.0 in |
| Check: 3 in AC on 30 in base | adequate True -> False (SN over the base) |
| Figure 3.1 printed inputs | unchanged |
| Rigid, CTB subbase Esb 400,000, MR 5,000, Dsb 6 | ValueError (reference k about 3,000) -> k 487, slab 9.0 in |
| Rigid, Esb 50,000, Dsb 18, LS 1 | k 141 -> 168 (slab 9.5 both) |
| Figure 3.3 printed example | k 390 -> 391 |
| PAV-6 comparison, AASHTO side | 13.5 -> 16.0 in (base modulus from Fig 2.6) |
| PAV-6 comparison, UFC side | 9.5 -> 11.0 in |
| PAV-4 UFC flexible | required 6.9 -> 8.3 in; provided 10.5 (subbase minimum) |
| Appendix G-3 section | 8.0 -> 11.5 in (3.5 / 4 / 4) |
| PAV-5 UFC rigid, 650 psi, k 100, 300,000 passes | hd 5.7 -> 7.0, slab 6.0 -> 7.0 in |
| G-8.12, 650 psi, k 50, 1,200,000 | hd 6.5 -> 7.8, slab 6.5 -> 8.0 in |
| 400 psi, k 100, 1,000,000 | hd 6.6 -> 9.9, slab 7.0 -> 10.0 in |
| 900 psi, k 100, 1,000,000 | hd 6.2 -> 6.0, slab 6.5 -> 6.0 in |
| 800 psi, k 300, 10,000 | slab 4.5 -> 6.0 in (6 in minimum) |
| Eq 13-1, 650 psi, k 200, 5,000,000, Ef 650,000, hs 6 | slab 5.5 -> 6.0 in |
| G-7, 650 psi, k 100, 20,000,000 | 8.1 / 8.5 in unchanged |
| PAV-7 mixed traffic | 1,316,228 passes unchanged |

Eval keys re-derived by hand off the print (sources in the keys): PAV-4
8.3 (alt 8.5), PAV-5 7.0, PAV-6 AASHTO 16.0 and UFC 11.0 (alt 10.3). The
comparison worked example moved with them (16.5 / 11.0 / 10.3).

## REF candidates (reference layer, not changed here)

- Figure 3.3 fit (3,026 pci where the chart gives about 490; Dsb scaling).
- Figures 2.8 / 2.9 three-point tables (0.10 at the lowest tick; capped at
  0.22).
- Figure G.3 VR (2.5 to 4 times the chart).
- UFC Figure E-1 between anchors (up to 1.7 in under the curves).
- UFC Figure F-1 two-anchor model (barely moves with flexural strength;
  6.5 in at G-8.12's k 50 where the print says 8).
- Table 19-2 extensions and the Table 19-3 page.
- Figure 2.5 150,000 psi anchor (small).
- Figure 3.5's notes: its "chart reads" at k about 25 pci put the closed
  form up to 12.5 % off in D; a pixel read of the chart finds it within
  about 2 % for D 6-14 in, k 15-1,950 pci. The function is right; the notes
  overstate the error.

## Noted, not fixed

- The other Appendix E vehicle curves (E-2 to E-31) used by
  `ufc_mixed_traffic`, Figures 19-x (frost penetration, limited subgrade
  frost penetration), Figure 3.4's depth scaling, the LEF tables other than
  D.4, Figures 2.3/2.4 and `performance.py`'s Table 3.1 iteration (pinned
  by the reference layer's 63 tests): not checked: time, not source.
- UFC para 19-6.1's 10 % reduction for substantial fills and the limited
  subgrade frost penetration method are not offered; the module states the
  reduced-subgrade-strength method only.
- Between pass-level lines F-1 interpolates in log(passes); the lines are
  not evenly spaced in log, so a passes sweep shows small kinks at 30,000
  and 100,000.

## Owner questions

1. Rounding: the module rounds up to 0.5 in; the guide rounds flexible
   layers to the nearest 1/2 in and slabs generally to the nearest inch
   (Appendix H rounds 11.04 to 11). Keep rounding up?
2. The Esb/MR > 4 warning fires on the guide's own Table 3.3 (6.0 in
   March). Warn only (as now), or cap Esb at 4 MR for an unbound subbase?
3. The comparison now reads base/subbase moduli off AASHTO Figures 2.6/2.7
   instead of the FHWA subgrade correlation. A judgment call; agree?
4. The UFC 4 in subbase minimum (p. 20, "for most pavements") is applied
   whenever a subbase is given. Agree?

## Gates (targeted; own TMP; at >= 1000 MB free, one at a time)

- pavement_design/tests (82, 19 new: the Fig 2.8/2.9 scales, Section
  3.1.5, the layered check, the Fig 2.6/2.7 scales, Figure 3.3 anchors and
  the Esb/MR warning, Figure E-1 and F-1 Appendix G reads, G-3, the 6 in
  slab minimum and Table 10-1, PFS), validation_examples v055 and the new
  v074-v076 file: 103 passed.
- funhouse_agent test_pavement_design_adapter, test_worked_examples,
  test_nested_keys, test_top_level_params, test_method_aliases,
  test_calc_package_adapter, test_calc_package_assumed_inputs,
  test_reviewers, test_tool_usability_foundry_eval: 450 passed, 2 skipped.
- test_calc_package_tool_parity 176, test_adapter_ergonomics 170,
  test_choices_stated_defaults 5, test_gm179_unused_inputs 20,
  test_citations 5, test_reference_adapters 304, test_eval_suite_v54 10,
  calc_package/tests 264 (3 skipped): all passed.
- test_defaults_used: 792 passed, 14 skipped.
- webapp/tests/test_page_fingerprints.py 15 passed (no re-record);
  test_public_text 4 passed, and its checks run by hand on every new and
  changed file (the new files are untracked until commit): clean.
- Not run: the full suites. The manual is regenerated at merge.
