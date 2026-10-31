# pile_group against GEC-12 chapter 7 and UFC 3-220-20 chapter 6, as printed

BACKLOG GM-286, 2026-10-31. Read from the PDFs in `geotech-references/docs/`:
GEC-12 (FHWA-NHI-16-009) Vol I chapter 7 and Vol III Appendix D; UFC
3-220-20 (16 January 2025) chapter 6 and Appendix B. Page numbers are the
printed ones (GEC-12 Vol I: pdf index = page + 45; Vol III: index = page +
33; UFC: index = page + 37).

The module offers: the rigid-cap load distribution (the simple method on
vertical piles, and the 6-DOF stiffness method); the group efficiency factor;
block failure in clay; the lateral p-multiplier; group settlement by Meyerhof
(1976) and by the equivalent footing. It offers no group uplift method.

## Group efficiency factor (GEC-12 Secs. 7.2.2.1-7.2.2.2; UFC Table 6-26)

- **Printed.** Eq. 7-35 eta_g = R_ng / (n R_n) (GEC-12 p. 281). Cohesionless
  soil (p. 282): the sum of the individual resistances, where the group is not
  over a weak deposit and the piles are not jetted or predrilled; over a weak
  deposit, the lesser of the sum and block failure; no value printed for
  jetted or predrilled piles. Spacing of 3D recommended, AASHTO minimum 2.5D.
  Cohesive soil (p. 284): su < 2 ksf with the cap not in firm contact, 0.65
  at 2.5D to 1.0 at 6D, linear between; su < 2 ksf with the cap in contact,
  1.0; su > 2 ksf, 1.0; then the lesser of that and block failure. UFC Eq.
  6-37 (p. 431) R_r,g = min(n eta_g R_r, R_r,gblock), and Table 6-26 (p. 432)
  by pile type: driven piles as GEC-12; drilled shafts, coarse-grained with
  the cap not in contact 0.65 at 2.5, 0.80 at 3, 1.0 at 4, else 1.0; CFA,
  coarse-grained (either cap) 0.65 at 2.5 to 1.0 at 6, fine-grained 1.0; DDP,
  fine-grained su < 2 ksf with the cap not in contact 0.7 at 3 to 1.0 at 6,
  else 1.0; micropiles, the same row 0.65 at 2.5, 0.70 at 3, 1.0 at 6, else
  1.0. Minimum s/b: driven 2.5 (and 2.5 ft), shafts 2.5, CFA, DDP and
  micropiles 3.0. "Interpolation can be applied" (p. 431).
- **Module before.** Converse-Labarre, E_g = 1 - theta (n(m-1) + m(n-1)) /
  (90 m n), theta = atan(d/s). No held document prints it.
- **Module now.** `group_efficiency_factor` reads the printed row by soil,
  cap contact, su against 2 ksf (95.76 kPa) and pile type, linear between
  the printed points and held at the last one; it refuses s/b below the
  column's printed minimum. Where cap contact or su is left out and the row
  depends on it, the lower row is taken and the other given as the
  alternative, in `choices`. Converse-Labarre is removed.
- **Verdict.** Fixed to the print.

## Block failure (GEC-12 Eqs. 7-36, 7-37, Fig. 7-31; UFC Eqs. 6-38, 6-39)

- **Printed.** R_ng = 2D(B + Z) su1 + B Z su2 Nc (Eq. 7-36, p. 286), Nc =
  5[1 + D/(5B)][1 + B/(5Z)] <= 9 (Eq. 7-37). Fig. 7-31: B and Z to the outer
  faces of the outer piles. su1 the weighted average along the sides, su2 to
  2B below the toe; for cohesionless sides over a weak clay, the unit shaft
  resistance in place of su1 (p. 287). UFC Eq. 6-38 the same with fs1, su2
  over 2B to 3B, whichever is lower (p. 433); Eq. 6-39 the same Nc. Fig.
  6-15: Z the embedded depth, B <= L the plan dimensions (the p. 431 text
  says "B <= Z", a slip). Resistance factor: GEC-12 Table 7-1 (p. 224),
  "Block Failure, phi_bl | Cohesive | 0.60"; UFC Table 6-27 (p. 433, AASHTO
  2020): driven piles 0.60, drilled shafts 0.55, micropiles 0.60 (reduced
  by 20 % in marginal ground), each with fine-grained soil below the group.
  No held page prints one for CFA or drilled displacement piles. UFC App.
  B-5.4 takes B = L = 18 ft (p. 638, in Nc) for its 7x7 group of 1 ft piles
  at 3 ft: centre to centre of the outer piles, not the 19 ft to their
  outer faces (App. B-5.7, p. 641, gives its equivalent footing the same 18
  ft "based on the 7x7 square arrangement of piles on a 3-ft
  center-to-center spacing").
- **Module before.** One cu for both su1 and su2, Nc fixed at 9.
- **Module now.** `block_failure_resistance(B, Z, D, su1, su2)` by Eqs.
  7-36 and 7-37 (B taken as the smaller plan dimension), returning the side
  and base terms and Nc both as printed and as limited;
  `block_failure_capacity` takes the layout to the outer faces. The tool
  takes su1 and su2, or one cu for both, said in `choices`. With
  `pile_type` the factored resistance phi_bl R_ng is given by the printed
  factor (`block_failure_factored_kN`); for CFA and DDP groups it is null
  and the result says no held page prints one. The micropile 20 %
  reduction for marginal ground is stated, not applied. B and Z to the
  outer faces (Fig. 7-31), with UFC B-5.4's centre-to-centre block and its
  resistance given as the alternative in `choices`.
- **Verdict.** Fixed to the print. UFC Appendix B-5.4 pinned (V-060) at its
  printed 18 ft.

## Lateral p-multiplier (GEC-12 Sec. 7.3.7.6; UFC Table 6-38)

- **Printed.** AASHTO, as GEC-12 prints it (pp. 367, 370): 0.8, 0.4, 0.3 for
  rows 1, 2, 3 and higher at 3b; 1.0, 0.85, 0.7 at 5b; interpolate between.
  Table 7-23 lists lower soil-specific values to consider (p. 365). UFC Table
  6-38 (p. 466) extends it ("AASHTO (2020) guidance, extrapolated to a pile
  spacing of 6·b", p. 465): 1.0 for s/b > 6; and gives Mokwa (1999): 0.06
  s/b + 0.64, 0.11 s/b + 0.34, 0.14 s/b + 0.16, 0.16 s/b + 0.04 for rows 1,
  2, 3 and 4+, each <= 1. Interaction insignificant at 6b; closer than 3b
  not advisable (p. 465). Batter piles (GEC-12 pp. 396-397): qualitative
  only.
- **Module before.** The 3b values right, then a line to 1.0 at 5b for
  every row, printed nowhere (row 2 0.7 and row 3 0.65 at 4b), and the 3b
  values held below 3b; cited to a "GEC-12 Table 9-2" the 2016 edition does
  not have.
- **Module now.** `p_multiplier(row, s/b, method)`: 'aashto' interpolated
  3 to 5 to 1.0 at 6, then 1.0; 'mokwa' by the printed lines. Below 3b
  refused. Vol III's 0.90 / 0.625 / 0.5 at 4b and 1.0 / 0.85 / 0.7 at 5b
  (pp. 56, 137) and UFC Table B-32 pinned (V-060). The straight line from
  5b to 1.0 at 6b is the module's reading of p. 465's "extrapolated"; the
  sentence could also mean AASHTO's 3b-5b line carried on to 6b (rows 3+
  0.9 at 6b; rows 1 and 2 reach 1.0 by then). The tool's `choices` say so,
  with the other reading's value between 5b and 6b; the numbers are
  unchanged.
- **Verdict.** Fixed to the print.

## Group settlement, Meyerhof (1976) (GEC-12 Eqs. 7-50 to 7-52; UFC Eqs. 6-46, 6-47)

- **Printed.** S = 4 pf If sqrt(B) / N1(60) (Eq. 7-50, in, ksf, ft); silty
  sand 8 in place of 4 (Eq. 7-51); If = 1 - D'/(8B) >= 0.5, D' = 2/3 of the
  pile's embedded length (Eq. 7-52, p. 319). CPT form Eq. 7-53. UFC Eq. 6-46
  prints the form without the leading 4 and Eq. 6-47 If = max(1 - Z/(12B),
  0.5); for silty sand it says "doubled" and refers to an "Equation 6-51"
  that is the 2V:1H stress.
- **D', two readings.** Vol I says the embedded length; Vol III Table D-23
  takes 2/3 of DB, the embedment in the bearing stratum, and B and Z exterior
  edge to exterior edge. The module takes the depth it is given
  (`embedment_DB`) and its docstring states both readings.
- **Module now.** Unchanged numbers; `silty_sand` added (Eq. 7-51), and the
  coefficient and equation returned. All ten Table D-23 entries within
  0.45 % (V-060).
- **Verdict.** Agrees. UFC Eq. 6-46's missing 4 and its "doubled" pointer
  are print slips; the module follows GEC-12.

## Group settlement, equivalent footing (GEC-12 Sec. 7.3.5.3; UFC Sec. 6-5.8.4)

- **Printed.** Terzaghi and Peck's footing at D/3 above the toe, 2V:1H
  spread (Fig. 7-44 1H:2V; Eq. 7-55 delta_sigma = Q/((B + z)(Z + z))); Figs.
  7-45 and 7-46 and UFC Table 6-32 place it elsewhere for other profiles;
  elastic compression (Eq. 7-48; UFC Eq. 6-54, constrained modulus); to
  where delta_sigma is under 10 % of the footing's bearing stress (UFC p.
  449). GEC-12's own procedure (Sec. 7.3.5.3 step 1c, p. 324) and UFC's
  worked example (App. B-5.7, p. 641, "as recommended by FHWA (2016)") sum
  instead to where the increase is less than 10 % of the existing vertical
  effective stress. The GEC-12 text (p. 320) says "side slopes at 30 deg",
  which the figure and equation do not take.
- **Module before.** Influence to 5 Bg, printed nowhere.
- **Module now.** To the 10 % depth, (B + z)(L + z) = 10 B L, or to a given
  `influence_depth_m`; citations fixed; the uniform case only. UFC's rule is
  the default, as the owner ruled for downdrag (OWN-62); the result's
  `choices` name GEC-12's rule as the alternative, given as
  `influence_depth_m` (the function takes no effective-stress profile).
- **Verdict.** Fixed to the print.

## Rigid-cap load distribution (simple and 6-DOF)

- **Printed.** GEC-12 Vol III Fig. D-25's free body and Tables D-13, D-14
  (pp. 47-49): the rows share Q and Muy as a rigid cap on vertical piles
  does; Vuy shared equally. EM 1110-2-2906 and CPGA (ITL-89-4) are not
  held, and no held document prints a 6-DOF or battered group example.
  Converse-Labarre is printed in no held document either.
- **Module.** Unchanged. The simple method gives 2908.75 / -403.75 kips
  (printed 2908 / -406, a slip: -403.75 by its own loads), 2069 / -231, and
  1810.5 / -270.5 (printed 1811 / -271); per pile 323.2 / 264.4 / 223.8
  (printed 323 / 265 / 224); Table D-14's -46 kips against -44.9. The 6-DOF
  method with equal springs gives the same, and Vuy 47.0 / 38.5 / 32.5 kips
  per pile (printed 47 / 39 / 33).
- **Verdict.** Agrees, pinned on the vertical-pile case only (V-060).

## Not offered

- Group uplift (GEC-12 Sec. 7.2.3.2, Eqs. 7-39 to 7-41, Fig. 7-33, the
  1H:4V soil block, pp. 290-292; phi_ug 0.50 in Table 7-1).
- The CPT group settlement (Eq. 7-53).

## Before and after

196 recorded calls before (the pile_group, calc package, funhouse group and
validation tests), 301 after. Of 102 distinct recorded inputs: 61 the same,
22 moved, 19 not called after. The rigid-cap methods (115 calls), block
failure (Nc above 9 in the recorded case) and the Meyerhof numbers did not
move (Meyerhof only adds `coefficient` and `equation`). Moved: the raft
settlement (influence depth 5B to the 10 % depth: 3x3, 1.5 m, 15 m, 2000 kN,
30 MPa from 16.45 to 13.74 mm; with one sublayer the hand case rises, 11.34
to 13.87 mm); the p-multiplier at 5b, rows 2 and 3, 1.0 to 0.85 and 0.70.
Not called after: Converse-Labarre (17 inputs); the calc package now gives
eta_g only with `soil_type`, and null without it.
