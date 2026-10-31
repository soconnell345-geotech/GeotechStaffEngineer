# drilled_shaft against GEC-10 chapters 8 and 10, as printed

BACKLOG GM-284, 2026-10-31; it takes in GM-27's flags. Read from the PDFs in
`geotech-references/docs/`: GEC-10 (FHWA-NHI-18-024, 2018) chapter 8
(Table 8-4), chapter 10 and Appendix A; FHWA NHI-06-089 (2006) Vol 2
ch. 9 for the named options; UFC 3-220-20 (2025) where it bears. Page
numbers are the printed ones (GEC-10: pdf index = page 10-n at 279 + n,
p. 8-14 at 221, p. A-11 at 593; NHI-06-089 Vol 2: p. 9-129 at 238).

The methods the module offers: side resistance in cohesionless soil (the
GEC-10 beta, Eq. 10-13, default; the depth beta of O'Neill and Reese as an
option), in cohesive soil (Eq. 10-18, default; the AASHTO alpha as an
option), in rock (Eq. 10-21; Eq. 10-22 added here); base resistance in
cohesionless soil (Eq. 10-14), cohesive soil (Eqs. 10-19, 10-20 with Table
10-2, default; O'Neill and Reese as an option) and rock (Eq. 10-23; the
Hoek-Brown base, Eqs. 10-24 to 10-29; the RQD reduction as an option); the
LRFD factors of Table 8-4. It offers no IGM method, no uplift run, no load
test factors and no settlement (Sec. 10.3.7); those are not checked here.

## Side resistance, cohesionless soil (Sec. 10.3.5.1, pp. 10-9 to 10-14)

- **Equations.** R_SN = pi B dz f_SN (Eq. 10-3), f_SN = beta sigma'v (Eq.
  10-5, sigma'v "average vertical effective stress over the depth
  interval"), beta = K tan delta (Eq. 10-6), Ko = (1 - sin phi') OCR^(sin
  phi') (Eq. 10-8), OCR = sigma'p / sigma'v (Eq. 10-9), Ko < Kp (Eq.
  10-10), sigma'p / pa = 0.47 N60^m with m 0.6 (clean sand) or 0.8 (silty
  sand to sandy silt) (Eq. 10-11), 0.15 N60 for gravel (Eq. 10-12),
  together Eq. 10-13. The module matches, on each layer's average sigma'v.
- **Limits.** "The value of beta at shallow depths should be limited to the
  value corresponding to a depth of 7.5 ft" (pp. 10-11, 10-12): the module
  takes sigma'v in the OCR no lower than the sigma'v at 7.5 ft, which is
  the same thing, beta falling with sigma'v. No limit on f_SN is printed;
  the module applies none.
- **K/Ko.** 1 by default (p. 10-11), 2/3 or lower where construction is not
  controlled (p. 10-12): the module's `K_over_Ko`.
- **Example 10-1 (pp. 10-12 to 10-14).** Pinned (V-058): beta 0.967 /
  0.976, f_SN 835 / 2,233 psf, 828.1 kips against the printed 0.97 / 0.98,
  837 / 2,248 psf, 833 kips; the gap is the example's rounding of OCR and
  Ko, and taken the same way Eq. 10-13 gives 833.
- **The depth beta (option).** NHI-06-089 Eq. 9-42 (p. 9-132), beta = 1.5 -
  0.135 sqrt(z, ft), 0.25 to 1.2, and "The limiting value of fsi for
  shafts in cohesionless soils is 4 ksf (191 kPa)" (p. 9-133). The module
  matches. **GM-27's "fs cap uncited"**: it is cited and printed there; it
  belongs to the option only. The N60/15 reduction below N60 15 is O'Neill
  and Reese (1999), not held; the result says so.
- **Casing.** p. 10-12: casing reduction factors 0.60 to 0.75 are commonly
  used (AASHTO 2007a), or side resistance "may be neglected" where casing
  is specified in soft soils or scour zones. The module neglects it over
  `casing_depth`; the reduction factors are not built.
- **Resistance factor.** 0.55 (p. 10-12; Table 8-4). Matches.

## Side resistance, cohesive soil (Sec. 10.3.5.2, pp. 10-15 to 10-18)

- **Equations.** R_SN = pi B dz alpha su (Eq. 10-15); su(UC)/su(CIUC) =
  0.893 + 0.513 log(su(UC)/sigma'vo) (Eq. 10-16), 0.911 + 0.499 log for UU
  (Eq. 10-17); alpha = 0.30 + 0.17 / (su(CIUC) / pa) (Eq. 10-18). The module
  matches, sigma'vo at the middle of the part carrying side resistance.
- **Top and bottom.** "neglect side resistance over the top 5 ft ... or to
  the depth of seasonal moisture change" (p. 10-17), printed in this
  section only: the module leaves out 1.5 m in cohesive soil, none in sand
  or rock (GM-244). No bottom zone is printed; Example 10-2 takes the clay
  to the base. The bottom 5 ft is NHI-06-089 p. 9-129, the module's
  `bottom_exclusion` option.
- **Example 10-2 (p. 10-18).** Pinned: su 2,000 psf UU, sigma'vo 3,697 psf,
  su(CIUC) 2,571 psf, alpha 0.44, f_SN 1,131 psf, R_SN 479.7 kips; the
  module gives each.
- **The AASHTO alpha (option).** NHI-06-089 Eqs. 9-37a/b (p. 9-129): 0.55
  to su/pa 1.5, 0.55 - 0.1 (su/pa - 1.5) to 2.5. Matches to 2.5; the
  module carries the line on to 0.35 and holds it, which nothing held
  prints (the result says so).
- **Casing.** p. 10-17: reduction 50 to 75 % for cased shafts in clay, no
  data to support it; not built (casing neglects it).
- **Resistance factor.** 0.45 (Table 8-4). Matches.

## Side resistance, rock (Sec. 10.3.5.3, pp. 10-20 to 10-22)

- **Eq. 10-21.** f_SN / pa = C sqrt(qu / pa), C = 1.0 for a "normal"
  socket (p. 10-20), the module's default (GM-244); C = 1.9 for roughened
  sockets verified by load tests (p. 10-22). The print says "with use of
  Equation 10-22" for the 1.9; Kulhawy and Prakoso's C is in Eq. 10-21's
  form, as the module takes it.
- **qu against the concrete. Fixed.** "For design, the value of qu used in
  Equation 10-21 should not exceed the compressive strength of the drilled
  shaft concrete", unless load tests show more (p. 10-21). The module took
  qu as given and refused `concrete_fc` (GM-160: it changed nothing). Now
  `concrete_fc` (kPa) limits qu in the rock side resistance, in Eq. 10-22
  too (NHI-06-089 Eq. 9-47, p. 9-140, limits that form by 0.65 pa (f'c /
  pa)^0.5; UFC 3-220-20 p. 430 takes "the lesser of" the two); left out,
  the result says the limit was not checked. The base is not limited by
  it (p. 10-25's check of the base against the concrete is structural,
  Ch. 12, not built). Example 10-4 checks it: qu 360 ksf against f'c 576
  ksf, no limit binds; pinned.
- **Eq. 10-22 with Table 10-3. Added.** f_SN / pa = 0.65 alpha_E sqrt(qu /
  pa), alpha_E from Table 10-3 by RQD and joint condition, "where caving or
  instability of the socket excavation is anticipated" (p. 10-21). The
  module had it only as C = 0.65 with alpha_E = 1.0. Now
  `rock_side_method="caving"` with `rock_joints` ("closed" default,
  "open"). Table 10-3 read on the page image (the text layer drops the
  repeated cells): closed 1.00, 0.85, 0.60, 0.50, 0.45 and open or
  gouge-filled 0.85, 0.55, 0.55, 0.50, 0.45 at RQD 100, 70, 50, 30, 20.
  UFC 3-220-20 Table 6-25 (p. 430, after FHWA 1999) prints the same
  alpha_E. GEC-10 says nothing of values between the rows (linear taken)
  or below RQD 20 (held at 0.45); the result says both.
- **Top of the socket.** No top zone in rock is printed (Sec. 10.3.5.3;
  Example 10-4 takes the socket's whole 10 ft). **GM-27's "rock socket
  drops the top 1.5 m even at ground surface"**: fixed by GM-244 (`_top_zone`
  gives 0 in rock unless `top_exclusion` is given); pinned in
  `test_gec10_2018_printed.py::test_rock_from_the_top_and_its_base_choice`.
- **Example 10-4 (pp. 10-27, 10-28).** Pinned: f_SN 27.6 ksf, R_SN 5,202
  kips. The print's "qu = 300 ksf" in the f_SN line is a slip; it computes
  with 360.
- **Resistance factor.** Table 8-4: 0.50 for Eq. 10-21 (0.40 uplift), no
  row for Eq. 10-22; p. 10-21's text gives 0.55 for both. The module takes
  the table's 0.50 (owner, GM-278) for both; the result's basis says so
  under Eq. 10-22.

## Base resistance, cohesionless soil (Eq. 10-14, pp. 10-14, 10-15)

- **Equation.** q_BN (tsf) = 0.60 N60 < 30 tsf; limited to N <= 50, above
  which load testing, else the upper bound. The module: 57.5 N60 kPa
  (0.60 tsf = 57.46 kPa, so 0.08 % high; noted below), N60 limited to 50.
- **N60. Fixed.** "N60 is the average value between the base and two
  diameters beneath the base" (p. 10-15); Appendix A takes it so ("the
  average field N-value (uncorrected) over a depth of two shaft diameters
  beneath the tip", p. A-11; "Mean N-value for 2-Diameters Below Tip 41",
  Table A-4, p. A-12). The module took the tip layer's N60 alone. Now the
  mean by thickness of the cohesionless layers' N60 from the base to 2B
  below it (B the base's diameter), limited to 50 after the mean; a layer
  there with no N60, or of clay or rock, is left out and named; a profile
  ending short of 2B gives the mean over what it has, as the clay base
  does (GM-252). The tip layer's N60 alone is the choice's alternative
  with its values.
- **sigma'v.** "the N-value to be used in Equation 10-14 is the field
  value, with no correction for overburden stress, although it is
  recommended that N-values correspond to 60 percent hammer efficiency"
  (p. 10-15). **GM-27's "sand tip independent of sigma'v"**: printed, not
  a fault; the result's basis now quotes it.
- **Alternatives.** Bearing capacity theory is allowed, Eq. 10-14
  "recommended for routine design" (p. 10-15); not built. No large-base
  reduction is printed (O'Neill and Reese's 1.27/B stays an option, off).
- **Example.** Appendix A (V-008): N60 41, 49.2 ksf, 2,473 kips. Pinned.
- **Resistance factor.** 0.50 (p. 10-15; Table 8-4). Matches.

## Base resistance, cohesive soil (pp. 10-19, 10-20)

- **Equations.** q_BN = N*c su (Eq. 10-19), su the mean over 2B below the
  base; N*c 9.0 where D >= 3B and su >= 2,000 psf, else Table 10-2 (500 /
  1,000 / >= 2,000 psf: 6.5 / 8.0 / 9.0, linear between); (2/3)[1 + D/(6B)]
  N*c su where D < 3B (Eq. 10-20). The module matches (GM-244, GM-252);
  below 500 psf N*c is held at 6.5 and said.
- **Example 10-3 (p. 10-20).** Pinned: 9.0 x 2,571 psf = 23.1 ksf, 453.6
  kips (the module 454.3, the print's rounding of 23.14 ksf).
- **O'Neill and Reese (option).** NHI-06-089 Eqs. 9-39, 9-40 (p. 9-131):
  Nc = 6[1 + 0.2 z/D] <= 9, 80 ksf, x 0.67 where the soil within 2D has su
  < 0.5 ksf. Matches; the 0.67 is taken on the tip layer's cu, which the
  result says.
- **Resistance factor.** 0.40 (p. 10-19; Table 8-4). Matches.

## Base resistance, rock (pp. 10-22 to 10-25)

- **Eq. 10-23.** q_BN = N*cr qu, N*cr = 2.5 under three conditions (massive
  or tightly jointed to one diameter below the base; no cavities; a clean
  base). The module matches; whether they hold is the engineer's
  (`rock_base_conditions_met`, GM-244). NHI-06-089 p. 9-141 prints the
  same 2.5 qu with one more condition, a socket deeper than 1.5 diameters;
  GEC-10 (2018) does not print it, and the module follows GEC-10.
- **Hoek-Brown base, Eqs. 10-24 to 10-29.** Read again on pp. 10-23,
  10-24: m_b / m_i = exp((GSI - 100)/28), s = exp((GSI - 100)/9), a = 1/2
  + (1/6)(exp(-GSI/15) - exp(-20/3)), A = sigma'vb + qu (m_b sigma'vb / qu
  + s)^a, q_BN = A + qu (m_b A / qu + s)^a, N*cr limited to 2.5. The module
  matches (GM-251); Fig. 10-8 pinned within 0.02.
- **The RQD reduction (option). GM-27's "RQD schedule uncited"**: no held
  document prints 2.5 / 1.5 / 0.8 / 0.4 at RQD 70 / 50 / 25 / 0. All held
  PDFs were searched for a page with RQD, a base resistance and those
  values; NHI-06-089 p. 9-141 gives only 2.5 qu and refers jointed rock
  to FHWA (1999), and UFC 3-220-20 p. 431 gives N*cr 2.5 and otherwise
  testing, the Hoek-Brown criterion or no base. The module already says
  "not checked against a printed page" and keeps it off by default.
- **Example 10-4.** Pinned: 900 ksf on the 6 ft socket, 25,447 kips, phi
  R_BN 12,723 kips.
- **Resistance factor.** 0.50 for Eq. 10-23 or 10-29 (p. 10-22, p. 10-24;
  Table 8-4). Matches.

## Resistance factors (Table 8-4, p. 8-14) and Example 10-5

- Side: 0.55 sand (beta), 0.45 clay (alpha), 0.50 rock (Eq. 10-21); uplift
  0.45, 0.35, 0.40. Base: 0.50 sand (Eq. 10-14), 0.40 clay, 0.50 rock. The
  module's `RESISTANCE_FACTORS` match. Not built: the cohesive IGM row
  (0.60 / 0.50, no IGM method), load tests (0.70 compression, 0.60
  uplift), groups, structural factors, the extreme-event uplift 0.80
  (p. 10-39). Uplift takes no base and treats the shaft's weight as a load
  (Eq. 10-32, p. 10-39); `apply_lrfd` gives compression only.
- **Example 10-5 (p. 10-29). Pinned.** 0.55 (197.2 + 635.6) + 0.45 x 479.7
  + 0.40 x 453.6 = 855.4 kips (the text also prints 890.1, a slip); the
  module 853.0 kips, 0.3 % under, from Example 10-1's rounding.

## Values that move (old -> new; hand-checked against the print)

No pinned value moves: every test, validation example (V-006, V-007,
V-008, V-058), eval key (DS-1, DS-2) and worked example (WE-SHAFT-1) gives
what it gave. None has a sand base with other soil within 2B below it,
rock with a concrete strength below its qu, or caving rock. The changes
reach these inputs only:

| Input | Old | New |
|---|---|---|
| sand base, N60 differing within 2B (made-up: base 9.5 m, B 1 m; N60 20 to 10 m, 40 below) | N60 20, q_b 1,150 kPa, Q_b 903 kN | N60 (0.5 x 20 + 1.5 x 40)/2 = 35, q_b 2,012.5 kPa, Q_b 1,581 kN |
| rock side, f'c below qu (made-up f'c 10,000 kPa on Example 10-4's socket) | f_s 1,321.6 kPa (qu 17,237) | f_s sqrt(10,000 x 101.325) = 1,006.6 kPa |
| rock side, `rock_side_method="caving"` (made-up RQD 60, closed joints, qu 8,000 kPa) | not offered (C 0.65 gave alpha_E 1.0: 585.2 kPa; the default C 1.0 gives 900.3) | 0.65 x 0.725 x sqrt(8,000 x 101.325) = 424.3 kPa |

Test text that moved with them: the rock C choice's 0.65 entry now reads
"the value of Eq. 10-22 with alpha_E = 1.0"
(`test_rock_socket_example.py`), and GM-160's refusal test became
`test_the_tools_take_the_concrete_strength_for_the_rock_side`.

## Noted, not fixed

- The top zone in clay is 1.5 m; the print's 5 ft is 1.524 m. 0.024 m more
  clay side resistance (under 0.3 % of a typical shaft's); `top_exclusion=
  1.524` gives the print. The DS-1 eval key derives with 1.5.
- 57.5 kPa a blow in Eq. 10-14 is 0.60 tsf rounded (57.46): 0.08 % high.
- The rock side takes pa = 101.325 kPa whatever the `pa` input (which the
  tool documents for the rational chains only).
- The calc steps' rock note for a C other than 1.0 names Eq. 10-22's 0.65
  even for C = 1.9.
