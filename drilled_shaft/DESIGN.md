# drilled_shaft — Drilled Shaft Axial Capacity

## Purpose
Computes axial capacity of drilled shafts (bored piles) using GEC-10 methods:
alpha method (clay), beta method (sand), and rock socket design.

## References
- FHWA GEC-10 (FHWA-NHI-18-024, 2018), Chapter 10 and Table 8-4; the 2010 edition (FHWA-NHI-10-016) is cited by name where a step follows it
- O'Neill & Reese (1999) as FHWA NHI-06-089 (Sec. 9.12) prints it — the
  older methods, kept as named options (`alpha_method="aashto"`,
  `beta_method="depth"`, `clay_base_method="oneill_reese"`,
  `bottom_exclusion`, `large_base_reduction`)
- AASHTO LRFD Bridge Design Specifications — resistance factors

## Files
- `shaft.py` — DrillShaft geometry (diameter, socket, bell)
- `soil_profile.py` — ShaftSoilLayer and ShaftSoilProfile
- `side_resistance.py` — alpha_method(), beta_method(), rock_socket()
- `end_bearing.py` — clay_Nc(), sand_N60(), rock_UCS()
- `capacity.py` — DrillShaftAnalysis orchestrator
- `lrfd.py` — AASHTO phi factors by method and condition
- `results.py` — DrillShaftResult with summary()/to_dict()

## Public API
```python
analyze_drilled_shaft(shaft, soil_profile, ...) -> DrillShaftResult
analyze_capacity_vs_depth(shaft, soil_profile, depths, ...) -> list
get_resistance_factors(method, condition) -> dict
```

## Key Notes
- ShaftSoilLayer dicts: soil_type/cu/phi/N60/qu/RQD (+ the GEC-10 beta's
  fields N1_60/sigma_v_ref/OCR/sigma_p_correlation)
- SoilProfile adapter: to_drilled_shaft_input(shaft_length)
- LRFD phi factors as GEC-10 Table 8-4 (p. 8-14) prints them: side 0.45
  clay, 0.55 sand, 0.50 rock (Eq. 10-21); base 0.40 clay, 0.50 sand and
  rock; uplift 0.35, 0.45, 0.40. The rock side was 0.55 until GM-244, from
  the text of p. 10-21 ('The resistance factor recommended with use of
  Equations 10-21 and 10-22 is phi = 0.55 ... as presented in Table 8-4'),
  which the table itself does not bear out; `lrfd_capacity` states that.
  The base factor follows the soil the base was computed in
  (`apply_lrfd` takes it from `result.base_steps` and refuses a
  `tip_soil_type` that differs; GM-244 review: the adapter's default
  "cohesive" factored a base in rock at 0.40). Table 8-4 prints no row for
  Eq. 10-22 (C = 0.65) or for the 'rqd' base; `lrfd_capacity` says so in
  its choices when either is used.
- A cohesionless tip needs its N60 (GM-142, 2026-10-18): q_b = 57.5 N60
  kPa (N60 capped at 50) has no other input, so `require_tip_N60` refuses a
  tip layer with none, in `compute`, the calc steps and
  `capacity_vs_depth` (which used to leave those depths out). Before, N60 =
  15 was taken without saying so. On the side, under the depth beta, a
  cohesionless layer with no N60 still gets no N60 < 15 reduction of beta
  (O'Neill & Reese); the GEC-10 beta (the default since GM-241) refuses it.
- The calc steps print the analysis's own end bearing (GM-148,
  2026-10-19): `end_bearing.unit_end_bearing_cohesive` /
  `unit_end_bearing_cohesionless` / `rock_end_bearing_multiplier` hold the
  arithmetic once, and both `compute` and the calc steps call them with the
  BASE diameter (the bell's where there is one). Before, the steps
  re-derived q_b with the shaft diameter and without the 3,830 kPa (80 ksf)
  limit or F_r. A side layer with no N60 under the depth beta is now a
  `choices` entry: NHI-06-089 Eq. 9-42 (p. 9-132) prints beta = 1.5 -
  0.135 sqrt(z, ft) with no N60 term; the N60/15 reduction is O'Neill &
  Reese (1999), which neither that manual nor GEC-10 (2018, rational beta
  only, Sec. 10.3.5.1, pp. 10-10 to 10-12) prints, so N60 is stated rather
  than required. `concrete_fc` is not an input of the tools (refused with
  the reason): the capacity never used it (GM-160).
- Rock not considered part of the socket (GM-220, 2026-10-20):
  `DrillShaft.fractured_rock_thickness` (m, default 0) is decomposed or
  fractured rock below the top of the first rock layer that GEC-10
  (FHWA-NHI-18-024) p. 8-15 lets the designer remove "but not considered
  part of the socket for design purposes". No rock side resistance is taken
  over it, through as many rock layers as it reaches; a soil layer keeps
  its own side resistance, and the base is the rock at the tip.
  `casing_depth` differs: it takes the side resistance of soil AND rock
  above it (GEC-10 Example 10-4, p. 10-27, cases the soil and the
  decomposed rock). Refused with no rock layer. A rock profile's result
  states the choice either way (`capacity.fractured_rock_choice`), and
  says so when the zone lies below the tip (it changes nothing) or holds
  the tip (the base takes that rock layer's qu and RQD as given; give the
  fractured rock as its own layer for the base on it); the calc
  steps print the zone and the side-resistance figure hatches it.
  `socket_diameter` stays one value for every rock layer: GEC-10 sizes a
  socket by one diameter B (Eq. 10-21; Example 10-4) and gives no socket
  diameter per rock layer.
- Rock socket base and side, GEC-10 Example 10-4 (GM-221, 2026-10-20):
  with the tip in rock the base is on `socket_diameter` (when given; a
  bell still governs), as Example 10-4 takes R_BN = pi/4 B^2 q_BN with B
  the 6 ft socket below 6.5 ft casing (p. 10-28); before, it was on the
  shaft diameter (29,865 kips for the example where GEC-10 prints 25,447).
  `DrillShaft.base_diameter(tip_in_rock)` / `base_area` hold the rule, the
  result carries `base_diameter_m` / `base_area_m2`, and the calc steps
  print A_tip on that diameter. The rock side resistance's C is
  `rock_side_C` (f_s = C pa sqrt(qu/pa)): default 1.0 since GM-244, Eq.
  10-21 as GEC-10 recommends it for routine design of a "normal" socket
  (pp. 10-20, 10-21); 0.65, the default before, is Eq. 10-22 (O'Neill and
  Reese 1999, where caving is anticipated) with alpha_E = 1.0 (give 0.65
  alpha_E by Table 10-3 for lower RQD or open joints), and C = 1.9 is for
  roughened sockets verified by load tests or local experience (p. 10-22).
  The result states the choice with each alternative's numbers
  (`capacity.rock_side_choice`). The tools reproduce Example 10-4 by
  default: R_SN 5,202 kips, R_BN 25,447 kips, phi R_BN 12,723 kips
  (`tests/test_rock_socket_example.py`).

## GEC-10 (2018) as printed is the default (GM-241, GM-244)

Every step follows GEC-10 (FHWA-NHI-18-024) as printed by default; the
older method stays as a named option with its source, and the result's
`choices` state the method, the step, the page, given vs default and the
alternatives with their values. Where the manual leaves a call open the
default is the printed one and the choice's basis says it is the engineer's
judgment. GM-244 (2026-10-20) moved these:

| Step | Default (2018 as printed) | Before GM-244 (the named option, where there is one) |
|---|---|---|
| clay side | Eq. 10-18, alpha = 0.30 + 0.17/(su(CIUC)/pa), p. 10-16 | `alpha_method="aashto"`: NHI-06-089 Eqs. 9-37a/b, 0.55 to su/pa 1.5 |
| clay, bottom of the shaft | side resistance to the base (Example 10-2, p. 10-18) | `bottom_exclusion` (m): NHI-06-089 p. 9-129, the bottom 5 ft; Fig. 9-49 one diameter |
| clay base | Eq. 10-19 with Table 10-2's N*c (p. 10-19), Eq. 10-20 where D < 3B; su the mean of the cohesive soil over 2B below the base (p. 10-19, GM-252), the tip layer at its su(CIUC) where cu is a UC/UU strength (Example 10-3) | `clay_base_method="oneill_reese"`: NHI-06-089 Eqs. 9-39, 9-40, Nc = 6[1 + 0.2 z/D] <= 9, x 0.67 below 0.5 ksf, q <= 80 ksf |
| rock base | Eq. 10-23, N*cr = 2.5 (p. 10-22; Example 10-4 with RQD 37 to 100), where its three conditions hold; else the Hoek-Brown base, Eqs. 10-24 to 10-29 (GM-251): see below | `rock_base_method="rqd"`: q_b/q_u reduced below RQD 70 (cited to the 2010 edition, not in the reference library; Table 8-4 gives no factor for it) |
| rock side C | 1.0, Eq. 10-21 for a "normal" socket (pp. 10-20, 10-21) | `rock_side_C=0.65`: Eq. 10-22 with alpha_E = 1.0 |
| large base | none (Eqs. 10-14, 10-19, 10-20 print none) | `large_base_reduction=True`: O'Neill & Reese 1.27/B in sand (B > 1.27 m), F_r in clay (B > 1.90 m) |
| top zone in rock | none (Sec. 10.3.5.3; Example 10-4 takes the socket's whole length) | `top_exclusion=1.5` |
| phi' of sand | the layer's phi' where given; 27.5 + 9.2 log (N1)60 only where none is (a layer may give (N1)60 alone) | (N1)60 overrode a given phi'; no option, the choices give the correlation's value |
| depth beta's limit | 4 ksf = 191.5 kPa (NHI-06-089 p. 9-133) | 200 kPa; no option |
| rock side resistance factor | 0.50 (Table 8-4) | 0.55 (the text of p. 10-21); no option, `lrfd_capacity`'s choices give it as the alternative |
| base resistance factor | by the soil the base was computed in | the adapter's `tip_soil_type` default "cohesive" (0.40), whatever the tip |

The rock base's conditions (GM-244 review). Eq. 10-23 is recommended
where the rock below the base is massive or tightly jointed to at least
one diameter below it, with no solution cavities or voids and a clean base
(p. 10-22); for a fractured rock mass GEC-10 gives the Hoek-Brown base
(Eqs. 10-24 to 10-29). Whether the conditions hold is the engineer's
judgment, the call's `rock_base_conditions_met`: true takes 2.5 qu as
stated (Example 10-4 does so with RQD 37 to 100 because "Rock coring and
local experience indicate the rock mass below the socket tip is
competent"); false takes the Hoek-Brown base, refused where the tip rock
layer gives no `GSI` or `m_i`; left out, the Hoek-Brown base where the tip
rock layer gives `GSI` and `m_i`, else the conditions are taken as met and
said so when every layer within one base diameter below the base is rock
with RQD 100 (the layer default), and the call is refused, naming GSI and
m_i, when any is soil or has a lower RQD, so 2.5 qu is never applied
silently to fractured rock. No RQD threshold is built: GEC-10 prints none.

The Hoek-Brown base (GM-251) is built as printed: Eq. 10-24 (p. 10-23),
Eqs. 10-25 to 10-29 (p. 10-24), on the tip rock layer's `qu`, `GSI` and
`m_i` and sigma'vb, the vertical effective stress at the base, with N*cr =
q_BN/qu limited to 2.5 (p. 10-24, "unless local experience or load tests
can be used to validate higher values"; the choices give the unlimited
value). GEC-10 names GSI only (not RMR) and points to GEC 5 Table 9-10 for
m_i, so both are the engineer's inputs. No GEC-10 worked example uses
these equations; the check is Fig. 10-8 (p. 10-25, sigma'vb = 0, m_i 4 to
33), whose plotted curves the equations match within 0.02 in N*cr from
GSI 30 to 95 (`tests/test_hoek_brown_base.py`). GEC-10 gives no depth
below the base over which the rock mass is taken: the tip layer's GSI and
m_i are used, and the choices list each other rock layer within one base
diameter below the base on its own. Resistance factor 0.50: p. 10-24 ("The
recommend resistance factor for application to Equation 10-29 is phi =
0.50") and Table 8-4 (p. 8-14, "Eq. 10-23 or Eq. 10-29"), which agree.

The clay base's su (GM-252). p. 10-19 defines su in Eqs. 10-19 and 10-20
as the "mean undrained shear strength of the cohesive soil over a depth of
2B below the base". The tool takes it by thickness over the base to 2B
below it (B the base's diameter, the bell's where belled): the tip layer
at its su as its side resistance took it, each cohesive layer below at its
cu, made su(CIUC) at the middle of its part within 2B where cu is a UC or
UU strength. A cohesionless or rock layer within 2B is left out (it has no
su), and the profile is not carried on below its last layer: where it
ends short of 2B the mean is over what it gives, said in the choice. The
tip layer's su alone, this tool's su before GM-252, is the choice's
alternative. O'Neill and Reese's base (`clay_base_method="oneill_reese"`)
keeps the tip layer's cu.

Judgment calls the choices name: cu taken as su(CIUC) when `su_test_type`
is left out; N*c below 500 psf held at 6.5 (Table 10-2 starts there; the
line through its first two rows is the alternative); su(CIUC)/pa outside
Fig. 10-6's 0.25 to 5; whether the rock below the base meets Eq. 10-23's
three conditions; B in Eq. 10-20 taken as the bell's diameter; the
Hoek-Brown base's GSI and m_i taken as the tip rock layer's; where the
profile ends short of 2B below a clay base. Not built: the sand base's N60
averaged over 2B below the base.

## Side-resistance methods

`DrillShaftAnalysis` selects the side-resistance method per soil type. In
cohesionless soil the default is GEC-10 (2018) as printed (GM-241,
2026-10-20), and in cohesive soil since GM-244.

**Cohesionless — `beta_method`:**
- `"rational"` (default since GM-241) — GEC-10 (FHWA-NHI-18-024) Sec.
  10.3.5.1, Step 5 of the design for axial load (Sec. 10.3), pp. 10-10 to
  10-12, step by step (`DrillShaftAnalysis._gec10_beta`, `beta_gec10`):
  - `R_SN = pi B dz f_SN`, `f_SN = beta sigma'v` (Eqs. 10-5, 10-7), sigma'v
    the "average vertical effective stress over the depth interval" dz: each
    layer's average (`ShaftSoilProfile.average_effective_stress`, exact across
    the water table; at mid-layer for one unit weight, as Example 10-1 and
    Table 10-1 take it). No sub-layering by default: the choices give the
    value with 0.1 m sub-layers (p. 10-7 splits a layer only where its
    N-values vary by more than a COV of 45 %).
  - `phi'` the layer's where given, else `27.5 + 9.2 log10[(N1)60]` from
    `N1_60` (Example 10-1, p. 10-13; Appendix A Eq. A-2). A given phi' wins
    (GM-244); the choices give the correlation's value as the alternative.
  - `sigma'p` from N60 by the layer's `sigma_p_correlation`: `"clean_sand"`
    (default, Eq. 10-11 with m = 0.6), `"silty_sand"` (m = 0.8) or `"gravel"`
    (Eq. 10-12, `0.15 pa N60`) (`preconsolidation_stress_gec10`). A direct
    per-layer `OCR` bypasses this step.
  - `OCR = sigma'p / sigma'v` (Eq. 10-9), sigma'v the layer's average or its
    `sigma_v_ref` (pre-scour, Appendix A), and not less than the sigma'v at
    7.5 ft (2.29 m) in the profile: "The value of beta at shallow depths
    should be limited to the value corresponding to a depth of 7.5 ft"
    (pp. 10-11, 10-12).
  - `Ko = (1 - sin phi') OCR^(sin phi') <= Kp = tan^2(45 + phi'/2)` (Eqs.
    10-8, 10-10); `K = (K/Ko) Ko` with `K_over_Ko` = 1 by default ("a
    rational first-order approximation", p. 10-11), 2/3 or lower where
    construction quality is not controlled (p. 10-12).
  - `beta = K tan(delta)`, `delta = phi'` (Eqs. 10-6, 10-13). No upper limit
    on f_SN (GEC-10 prints none; a choice notes NHI-06-089's 4 ksf where f_SN
    passes 191 kPa).
  - Each cohesionless layer needs its N60 or OCR; without either the call is
    refused, naming the layer and `beta_method="depth"`.
  - The result's per-layer `beta_steps` and the calc package print each step
    with the layer's numbers; `choices` state the method, K/Ko, the sigma'p
    correlation, the top zone where it takes sand and the 7.5 ft limit where
    it binds, each with its alternatives and their values.
  - Validated: GEC-10 Example 10-1 (V-058, pp. 10-12 to 10-14, 828 vs 833
    kips) and Appendix A Layer 3 (V-006); the eval key DS-2 is a hand
    derivation by these steps (`funhouse_agent/tests/test_eval_keys_drilled_shaft.py`).
- `"depth"` — O'Neill & Reese (1999) as printed in FHWA NHI-06-089 Eqs. 9-41
  to 9-43: `beta = 1.5 - 0.135*sqrt(z_ft) = 1.5 - 0.245*sqrt(z_m)`, clamped
  to [0.25, 1.2], z the depth to the layer's centre, x N60/15 where N60 < 15;
  `f_s` limited to 4 ksf, 'The limiting value of fsi for shafts in
  cohesionless soils is 4 ksf (191 kPa)' (p. 9-133), taken as 4 x 47.88 =
  191.5 kPa (200 kPa until GM-244).
  `beta_cohesionless(z, N60)`. The default until GM-241; GEC-10 (2018) does
  not print it. (Unit-mixing defect FIXED 2026-07-19: the old code applied
  0.245 to feet; the calc package printed that formula until GM-241.)

**Top zone — `top_exclusion`:** left out, no side resistance over the top
1.5 m in cohesive soil, and side resistance from the top in cohesionless
soil and rock. GEC-10 p. 10-17 prints the top 5 ft in Sec. 10.3.5.2,
Cohesive Soils: "common design practice to neglect side resistance over the
top 5 ft of a drilled shaft, or to the depth of seasonal moisture change",
"subject to modification based on local experience and judgment of the
designer". Sec. 10.3.5.1 (cohesionless) prints no such zone, and Examples
10-1 and 10-5 take the sand's side resistance from the surface; Sec.
10.3.5.3 (rock) prints none either, and Example 10-4 takes the socket's
over its whole 10 ft below the casing (the rock's 1.5 m was the module's
carry-over until GM-244). Given, the zone applies in every layer (1.5
leaves the top 5 ft out of sand and rock too). The casing, where deeper,
takes its place. With sand or rock in the top 1.5 m the result states the
choice and Q_ult with the 1.5 m left out.

**Bottom zone — `bottom_exclusion`:** default 0, side resistance in
cohesive soil taken to the base, as Example 10-2 (p. 10-18) takes Layer 3's
over its whole 27 ft to the base; GEC-10 (2018) prints no bottom zone.
Given (m), no side resistance over that length above the base in cohesive
soil: NHI-06-089 p. 9-129 (O'Neill & Reese 1999) leaves out "the top and
bottom 5-ft of the shaft", its Fig. 9-49 the bottom one diameter (the
module's default until GM-244). The choices give both with their values.

**Cohesive — `alpha_method`:**
- `"rational"` (default since GM-244) — GEC-10 Eq. 10-18, Chen et al. (2011)
  `alpha = 0.30 + 0.17/(su_CIUC/pa)` (`alpha_cohesive_rational`), applied to
  the **CIUC-equivalent** strength, so `fs = alpha * su_CIUC`; validated on
  Example 10-2 (479.7 kips, `tests/test_gec10_2018_printed.py`). A UC/UU lab
  strength is first normalized by the Chen & Kulhawy (1993) transform
  (`su_to_ciuc`, selected by `su_test_type` = "ciuc" (default: cu taken as
  su(CIUC), a judgment the choices state) / "uc" / "uu"):
  - UC->CIUC: `su(UC)/su(CIUC) = 0.893 + 0.513*log10(su/sigma'v0)`  (Eq 10-16)
  - UU->CIUC: `su(UU)/su(CIUC) = 0.911 + 0.499*log10(su/sigma'v0)`  (Eq 10-17)
  - The GEC-10 Appendix A worked example applies the **UC** pair (0.893/0.513) to
    its mean strength, so `su_test_type="uc"` reproduces it.
- `"aashto"` — O'Neill & Reese (1999) as NHI-06-089 Eqs. 9-37a/b print it:
  `alpha = 0.55` for cu/pa <= 1.5, `0.55 - 0.1 (cu/pa - 1.5)` to 2.5; the
  module goes on along the line to 0.35 and holds it (NHI prints nothing
  above 2.5). `alpha_cohesive(cu)`. The default until GM-244.

Both GEC-10 chains are validated against the GEC-10 Appendix A design example
(FHWA-NHI-18-024, Steps 11.5-11.6) in `validation_examples/test_published_v006_v008.py`:
Layer 3 sand beta = 0.41 (RSN 470.7 kips) and Layer 4 clay alpha = 0.47
(RSN 368.1 kips), reproduced through the high-level path to within ~1%.

## References (rational chains)
- FHWA GEC-10 (FHWA-NHI-18-024), 2018 — Appendix A design example; Section 10.3.5
- Chen et al. (2011) — alpha adhesion regression (Fig 10-6)
- Chen & Kulhawy (1993) — UU/UC -> CIUC strength transforms (Eq 10-16/10-17)
- Mayne & Kulhawy (1982) — Ko = (1 - sin phi')*OCR^(sin phi')
- Mayne (2007) — sigma'p correlation to N60
