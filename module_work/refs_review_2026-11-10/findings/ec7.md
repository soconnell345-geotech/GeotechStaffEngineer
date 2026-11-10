# ec7: Eurocode 7 Parts 1 and 2, Caltrans Trenching and Shoring, FEMA P-2082 — findings register

Filed by the lead from reviewer `ec7`'s report messages (2026-11-10).
Packages: `eurocode_7_1` (31 functions), `eurocode_7_2` (43),
`california_trenching` (32), `fema_p2082` (12). "PDF p. N" is 1-based; the
printed page follows in brackets. EN 1997-1 and EN 1997-2: PDF page =
printed + 2. Evidence: `C:/dev/tmp/refs_review_2026-11-10/ec7/`
(`behaviour_out*.txt`, `img/`). Counts: **A 6, B 10, C 5.**

Coverage: all 118 functions read; every table, factor, coefficient, exponent
and sign print-checked (EN pages and every Caltrans/FEMA equation read off
rendered pages); about 330 calls through `dispatch.call_agent`. The July 2026
Annex A corrections and REF-52's EC7-2 fixes all hold; Tables A.1 to A.17
match cell by cell (142 values). Lead's own print check: ec7-01, ec7-03
confirmed.

## A

- **ec7-01** `eurocode_7_1/equations.py:82, :108` `active_earth_pressure`, `passive_earth_pressure`. Print PDF p. 143 (141): sigma = K[integral(gamma dz) + q - u] + u -/+ c Kc (C.1, C.2). Code omits "- u" inside the bracket. Kp 3.0, gamma 20, z 5, u 49.05: 349.05 vs 201.90 (passive 73 % high, unsafe); right only when u = 0.
- **ec7-02** `eurocode_7_1/tables.py:517-522` Table A.11 (PDF p. 136 (134)): columns are ">= 2, >= 5, >= 10, >= 15, >= 20"; code interpolates (n 4, min: 1.40 vs 1,50; unsafe). Table A.10 (`:458-462`, PDF p. 135 (133)) prints n = 1, 2, 3, 4, 5, 7, 10; code interpolates 6, 8, 9.
- **ec7-03** `california_trenching/tables.py:977-978` `factor_of_safety_requirements`: anchor block FS 1.5 citing Section 10-2; print PDF p. 213 (10-4): "a safety factor of 2 is recommended for all anchor blocks".
- **ec7-04** `eurocode_7_1/equations.py:286-298, :413-417, :430-436` Annex D: `shape="strip"` with no L is computed as a square (undrained +18.6 %, drained +10.3 %); a misspelt shape is silently rectangular; B > L accepted; the shape used is not returned.
- **ec7-05** `eurocode_7_2/tables.py:445` Table D.3 (PDF p. 117 (115); qc 10-25 MPa) and `:982` Table F.3 (PDF p. 128 (126); ID 40-100 %): below the first printed column the first column's value is returned (qc 4 MPa -> pb 2.0 MPa; ID 10 % -> 34 deg).
- **ec7-06** `california_trenching/tables.py:285` (unpacking `:318-328`) Table 3-1 "Very Loose" (PDF p. 43 (3-5)): print moist < 100 pcf, submerged < 60 pcf; code returns submerged 60 to 100 and no moist value.

## B

- **ec7-07** EN 1997-2 G.1 (`equations.py:517-528`; 3 <= N10 <= 50, PDF p. 131 (129)), G.3 (`:578-587`; four ranges, PDF p. 132 (130)), Eq. 4.5 (`:163-167`; IDMT < 0,8, PDF p. 59 (57)) answer outside the printed range.
- **ec7-08** 45 EN 1997 lookups return a bare number; `describe_method` drops the Returns text. Units a reader would not assume: Burland and Burbridge settlement in mm; `table_h_2_limiting_total_settlement` 0.05 is metres; `equation_k2_plt_modulus` MPa; `equation_g1_density_index_from_dp` a fraction while Tables F.1/F.3 take percent. (Library-wide: lead.)
- **ec7-09** Annex A: recommended values returned as the standard's (0 of 35 methods mention the National Annex); every selector has a default, so a call that leaves the set out gets the unfactored set; "applied to tan phi'" not in the result.
- **ec7-10** `design_approach_sets("DA1")` (`eurocode_7_1/tables.py:797-807`) returns Combination 1 alone; print PDF pp. 34-35 (32-33) requires both.
- **ec7-11** Selector aliases that pick one of two quantities: `"cohesion"` -> c' (`tables.py:93`), `"shaft"` -> shaft in compression (`tables.py:299`), Caltrans `"concrete"` -> mass concrete (`california_trenching/tables.py:702`).
- **ec7-12** Boolean parameters take any string as True (`"no"`, `"false"`): `eurocode_7_2/tables.py:1259, :446`; `california_trenching/tables.py:866, :129`; `eurocode_7_2/equations.py:461`. Best fixed once in dispatch.
- **ec7-13** Caltrans `max_allowable_slope_angle` (`equations.py:419-619`) solves Eqs. 4-5-1 to 4-5-4 (PDF pp. 73-74), which as printed are not dimensionally consistent: more cohesion gives a flatter slope between 400 and 800 psf. Safe side; misleading.
- **ec7-14** Printed footnotes that change the value do not reach the result: Tables D.1 and H.1 (E' 50 % lower in silty soil), D.5 notes, W.1's 0 count condition, F.3 (4) corrections.
- **ec7-15** FEMA: `seismic_design_category` (`fema_p2082/tables.py:382-383, :425`) skips the S1 >= 0.75 rule when S1 is not given (default 0.0); `site_class_from_vs30` (`:106-179`) leaves out Section 20.3's vs/1.3 and 1.3 vs rule.
- **ec7-16** Caltrans `at_rest_k0` (`equations.py:219-224`) echoes a slope it ignores when OCR > 1; `aep_cohesive_max_ordinate` (`:900-976`) defaults `factor` to 0.3 where the print gives 0.2 to 0.4 by judgment, and for 4 < Ns < 6 without `ka` returns one envelope where the print says use the larger. Related: OWN-67.

## C

- ec7-17 names citing what the print lacks (`table_h_1_*`, `table_h_2_*`: Annex H prints no table); the Kpc docstring caveat is wrong (the print gives Kpc); `importance_factor` cites a table P-2082 does not print.
- ec7-18 `pdf_page` 0-based in Caltrans and FEMA results; three printed-page slips (`california_trenching/equations.py:411, :65, :119`).
- ec7-19 `undrained_bearing_resistance`: q described as effective overburden (safe side).
- ec7-20 smaller items: `Infinity` in JSON from `heave_factor_of_safety`; phi rounded in `matrix_4_1_passive_reduction_factor`; `table_e2` key name; no sanity bounds in several EN 1997-2 equations; confusing error texts.
- ec7-21 package and adapter descriptions advertise lookups that do not exist (Bell's coefficients, FHWA lagging Table 6-2, Bjerrum-Eide Nc, Ovesen factor); three printed rules not coded (Eq. 8-3-4, Eq. 10-3-4, the hard layer within 0.7B).

## Patterns

P1 out of range answered: 9 functions. P2 bare float: 45 EN 1997 functions (Caltrans and FEMA return dicts with unit-named keys and a reference). P3 a selector left out or loosely named picks a row silently. P4 docstring-only caveats. P5 types not checked.
