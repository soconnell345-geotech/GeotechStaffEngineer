# How bearing_capacity combines its correction factors, against the print

BACKLOG GM-281, 2026-10-30. Read from the PDFs in
`geotech-references/docs/`: UFC 3-220-20 (2025) pp. 288-296 and Example B-3
pp. 618-623; FHWA GEC-6 pp. 53-63; EN 1997-1:2004 Annex D pp. 158-159;
FHWA NHI-06-089 Ch. 8 (it repeats GEC-6, Tables 8-4 to 8-6). Page numbers
are the printed ones.

## What the print says

- **UFC 3-220-20 Eq. 5-15 to 5-17 (p. 288):** q_ult = c N_c Psi_c +
  sigma'_zD N_q Psi_q + 0.5 gamma B N_gamma Psi_gamma (undrained saturated:
  s_u N_c Psi_c + sigma_zD Psi_q).
- **Eq. 5-18 to 5-20 (p. 293):** "In most cases, these factors are found by
  multiplying factors for individual effects": Psi_c = s_c d_c i_c b_c g_c,
  and the same for q and gamma. Terzaghi: "The other factors should be set
  equal to 1 in Eqn. 5-18 to 5-20" (p. 294).
- **Eq. 5-21 (p. 295), Brinch Hansen phi = 0:** "An important difference is
  that the lumped correction factor is additive":
  Psi_c = 1 + s_c + d_c - i_c - b_c - g_c, with Table 5-7: s_c = 0.2 (B/L),
  d_c = 0.4 k, i_c = 0.5 - 0.5 [1 - H/(A' C_a)]^0.5, b_c = eta/(pi/2 + 1),
  g_c = beta/(pi/2 + 1); Psi_q = 1.
- **Example B-3 (p. 620):** "the correction factors are usually multiplied
  with the exception of the undrained case for Brinch Hansen's method".
  Table B-16 (p. 621), Meyerhof undrained: Psi_c = [1 + 0.2(B/L)][1 +
  0.2(Df/B)]. Table B-17 (p. 622), Brinch Hansen undrained:
  Psi_c = [1 + 0.2(B/L)] + 0.4(Df/B).
- **p. 292:** "the correction factors cannot usually be applied across
  methods, i.e., Meyerhof corrections should be used only with the Meyerhof
  bearing capacity factors".
- **GEC-6 Eq. 5-14 (p. 63):** q_ult = c N_c s_c b_c + q N_q C_wq s_q b_q
  d_q + 0.5 gamma B_f N_gamma C_wgamma s_gamma b_gamma, multiplied, no d_c,
  no load inclination (Sec. 5.2.3.7, p. 62), sloping ground by N_cq and
  N_gamma-q (Eq. 5-11, p. 58). Table 5-2 (p. 56): phi = 0, s_c = 1 +
  B_f/(5 L_f). Table 5-5 (p. 60): phi = 0, b_c = 1 - alpha/147.3 (degrees).
- **EN 1997-1 D.1 (p. 158), undrained:** R/A' = (pi + 2) c_u b_c s_c i_c +
  q, multiplied, no depth factor. **D.2:** c' N_c b_c s_c i_c + q' N_q b_q
  s_q i_q + 0.5 gamma' B' N_gamma b_gamma s_gamma i_gamma, multiplied.

## Per method and case

| Method (`factor_method`) | Case | Printed combination | Module | Agree? |
|---|---|---|---|---|
| Vesic set (`vesic`) | phi > 0 | Multiplied: GEC-6 Eq. 5-14 (p. 63), EN 1997-1 D.2 (p. 158), and UFC Eq. 5-18 to 5-20 (p. 293) for Brinch Hansen's phi' > 0 factors (Table 5-8, p. 296), whose d_q and d_c the module takes | c N_c s_c d_c i_c b_c g_c + q N_q s_q d_q i_q b_q g_q + 0.5 gamma B' N_gamma s_gamma d_gamma i_gamma b_gamma g_gamma | Yes |
| Vesic set | phi = 0 | s_c, b_c (GEC-6 Eq. 5-14) and s_c, b_c, i_c (EN 1997-1 D.1) multiplied, no d_c in either. The module's d_c = 1 + 0.4 k is Brinch Hansen's Table 5-7 d_c = 0.4 k (p. 295), which Eq. 5-21 adds | Was c N_c s_c d_c i_c b_c g_c: d_c multiplied | **No, fixed:** now c N_c (s_c i_c b_c g_c + d_c - 1) |
| Meyerhof (`meyerhof`) | phi = 0 | Table 5-6 (p. 294) factors multiplied (Eq. 5-18); Example B-3 Table B-16 (p. 621) | c N_c s_c d_c i_c (b_c g_c); shape 1 under an inclined load (note A) | Yes (pinned to Example B-3, L = 10.84 ft) |
| Meyerhof | phi > 0 | Multiplied (Eq. 5-18 to 5-20); Table B-16 drained Psi_q = Psi_gamma = [1 + 0.1(B/L)K_P][1 + 0.1(Df/B)K_P^0.5] | Multiplied | Yes |
| Terzaghi (`terzaghi`) | any | Table 5-5 shape factors only, the rest 1 (p. 294): c N_c s_c + q N_q + 0.5 gamma B N_gamma s_gamma | Same | Yes |
| Brinch Hansen (1970) | any | UFC Tables 5-7 and 5-8, Eq. 5-21 | Not offered as a factor set: only his N-gamma (`ngamma_method="hansen"`, a mixed method), his k and depth factors, and GEC-6's base factors after him | n/a |
| Vesic (1973, 1975) own papers | any | Not in the library. UFC p. 288: Vesic's N factors "are nearly identical to those proposed by Meyerhof and Hansen". The module's Vesic set is GEC-6's (AASHTO 1996) with EN 1997-1's inclination factors; its phi = 0 i_c and its g factors cite Vesic (1975), not printed in the library (already stated in the choices and DESIGN) | n/a | Not checkable |

## The fix (Vesic set, phi = 0)

    c N_c (s_c i_c b_c g_c + d_c - 1),  d_c - 1 = 0.4 k

- With no depth factor (surface footing, `depth_factor_method` "gec6" or
  "none") it is the product GEC-6 and EN 1997-1 print, unchanged.
- With the shape factor alone it is Eq. 5-21 exactly (s_c, d_c are Table
  5-7's values), and reproduces Example B-3, Footing 1: L >= 9.97 ft printed
  (p. 623); module 9.969 ft now, 9.751 ft multiplied.
- With a base tilt, ground slope or load inclination as well, no source
  prints the whole combination. The module keeps their product (GEC-6, EN
  1997-1) and adds 0.4 k; Eq. 5-21 in full would add their decrements
  (b_c and g_c are numerically Table 5-7's; i_c is Vesic's, not Table
  5-7's). The choices basis says which form ran.

## Values that moved

| Where | Case | Old | New | Printed |
|---|---|---|---|---|
| `tests/test_factor_combination.py` (new) | UFC Example B-3 Footing 1, Vesic set = Brinch Hansen undrained | L = 9.751 ft | 9.969 ft | 9.97 ft (p. 623) |
| same | Meyerhof undrained | 10.847 ft | 10.847 ft | 10.84 ft (p. 622) |
| `tests/test_bearing_capacity.py::TestClayPhiZero::test_clay_phi0` | 2 m square, Df 1 m, s_u 50: c term | 370.08 kPa (q_ult 388.08) | 359.80 (q_ult 377.80) | no printed example for this footing; Eq. 5-21 by hand, 50 x 5.14 x (1 + 0.2 + 0.2) |
| `tests/test_depth_factor_method.py::test_df_b_one_before_and_after` | 2 m square, Df 2 m, s_u 50 | 441.29 kPa | 425.14 | Eq. 5-21 by hand, 50 x 5.14 x (1.2 + 0.4 pi/4) + 36 |
| `funhouse_agent/geotech_test_suite.json` BC-2 (eval key) | 3 m square, Df 1.5 m, s_u 75 | q_ult 582, q_all 194 | 567, 189 | Eq. 5-21 by hand, 566.7 / 188.9 |

No validation example moves (V-021 is phi = 35, c = 0). No worked example,
calc package fixture or reliability test in the repo uses phi = 0 with a
shape factor and depth.

## Noted, not fixed

- **Example B-3's own misprints.** Table B-16 prints d_c = 1 + 0.2(Df/B) =
  "1 + 0.2(3 ft)/L = 1 + 0.6/L" and Table B-17 d_c = "0.4(3 ft)/L = 1.2/L"
  and q_all,net = "1849 + 4934/L"; those give L = 11.45 and 10.85 ft. The
  printed L = 10.84 and 9.97 ft follow from Df/B = 3/6 = 0.5 (d_c = 1.1 and
  0.2), which the tests use.
- **The phi -> 0 step.** At phi = 0 the Vesic c term adds d_c; at phi > 0 it
  multiplies d_c (note C form, tending to 1 + 0.39 k). The step,
  about c N_c (s_c - 1)(d_c - 1), is the print's own: Brinch Hansen
  multiplies for phi' > 0 (Eq. 5-18) and adds for phi = 0 (Eq. 5-21).
- **GEC-6 Table 5-5 b_q = (1 - 0.017 alpha tan phi)^2, alpha in degrees.**
  The module uses alpha in radians (0.01745 per degree), which is EN 1997-1
  D.4's b_q = (1 - alpha tan phi)^2 (p. 158); the 0.017 is that rounded.
- **Meyerhof between phi = 0 and 10 deg.** Table 5-6 prints rows for phi = 0
  and phi' > 10 deg only; the module takes s_q = d_q = 1 (the phi = 0 row)
  with s_c, d_c at the actual K_P.
- **Shape and inclination together (Vesic set).** GEC-6 Table 5-2 note and
  Sec. 5.2.3.7 (p. 62) advise against applying both; EN 1997-1 D.1 and D.2
  multiply them, as the module does (already stated in DESIGN, GM-188).
- **Example B-3 drained checks are not pinned.** The Meyerhof drained
  length (L >= 3.32 ft) is shorter than B = 6 ft, and the Brinch Hansen
  drained one uses Table 5-8's s_q = 1 + sin phi' (B/L), not the Vesic set's
  GEC-6 s_q = 1 + (B/L) tan phi.

## Proposed BACKLOG item (changes what a live run returns)

- **Meyerhof with a base tilt or ground slope borrows another method's
  factors.** `factor_method="meyerhof"` multiplies GEC-6 Table 5-5's b
  factors and Vesic's (1975) g factors into Meyerhof's. UFC 3-220-20 Table
  5-3 (p. 291) excludes Meyerhof for a tilted base, p. 294 says "Meyerhof
  (1963) did not provide base or inclination factors in this form", and
  p. 292 says correction factors "cannot usually be applied across
  methods". Terzaghi already refuses these inputs. Either refuse (base tilt)
  and point to Sec. 5-3.4 (slope), or state the borrowing in the choices.
