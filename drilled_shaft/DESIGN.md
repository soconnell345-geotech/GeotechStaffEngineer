# drilled_shaft — Drilled Shaft Axial Capacity

## Purpose
Computes axial capacity of drilled shafts (bored piles) using GEC-10 methods:
alpha method (clay), beta method (sand), and rock socket design.

## References
- FHWA GEC-10 (Drilled Shafts)
- O'Neill & Reese (1999) — alpha method for clay
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
- LRFD phi factors: alpha=0.45, beta=0.55, rock=0.50 (redundant), etc.
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
  `rock_side_C` (f_s = C pa sqrt(qu/pa)): default 0.65, unchanged, which is
  GEC-10 Eq. 10-22 (O'Neill and Reese 1999, where caving is anticipated)
  with alpha_E = 1.0; GEC-10 recommends Eq. 10-21 with C = 1.0 for routine
  design of a "normal" socket (p. 10-21), 0.65 alpha_E by Table 10-3 where
  caving is anticipated, and C = 1.9 for roughened sockets verified by load
  tests or local experience (p. 10-22). The result states the choice
  (`capacity.rock_side_choice`). With C = 1.0 the tools reproduce Example
  10-4: R_SN 5,202 kips, R_BN 25,447 kips, phi R_BN 12,723 kips
  (`tests/test_rock_socket_example.py`). Which C is the default is the
  owner's call.

## Side-resistance methods

`DrillShaftAnalysis` selects the side-resistance method per soil type. In
cohesionless soil the default is GEC-10 (2018) as printed (GM-241,
2026-10-20); in cohesive soil the AASHTO alpha stays the default and GEC-10's
Chen (2011) alpha is opt-in.

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
  - `phi'` the layer's, or `27.5 + 9.2 log10[(N1)60]` where `N1_60` is given
    (Appendix A Eq. A-2).
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
  `f_s` capped at 200 kPa (NHI-06-089 prints 4 ksf = 191 kPa).
  `beta_cohesionless(z, N60)`. The default until GM-241; GEC-10 (2018) does
  not print it. (Unit-mixing defect FIXED 2026-07-19: the old code applied
  0.245 to feet; the calc package printed that formula until GM-241.)

**Top zone — `top_exclusion`:** no side resistance over the top 1.5 m by
default, in every layer: GEC-10 p. 10-17, "common design practice to
neglect side resistance over the top 5 ft of a drilled shaft, or to the
depth of seasonal moisture change", "subject to modification based on local
experience and judgment of the designer". It is printed for cohesive soils;
Example 10-1 takes sand from the surface (`top_exclusion=0`). The casing,
where deeper, takes its place.

**Cohesive — `alpha_method`:**
- `"aashto"` (default) — piecewise `alpha = 0.55` for cu/pa <= 1.5, decreasing
  above (floored at 0.35). `alpha_cohesive(cu)`.
- `"rational"` — GEC-10 Chen (2011) `alpha = 0.30 + 0.17/(su_CIUC/pa)`
  (`alpha_cohesive_rational`), applied to the **CIUC-equivalent** strength, so
  `fs = alpha * su_CIUC`. A UC/UU lab strength is first normalized by the Chen &
  Kulhawy (1993) transform (`su_to_ciuc`, selected by `su_test_type` =
  "ciuc"/"uc"/"uu"):
  - UC->CIUC: `su(UC)/su(CIUC) = 0.893 + 0.513*log10(su/sigma'v0)`  (Eq 10-16)
  - UU->CIUC: `su(UU)/su(CIUC) = 0.911 + 0.499*log10(su/sigma'v0)`  (Eq 10-17)
  - The GEC-10 Appendix A worked example applies the **UC** pair (0.893/0.513) to
    its mean strength, so `su_test_type="uc"` reproduces it.

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
