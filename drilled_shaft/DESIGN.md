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
- ShaftSoilLayer dicts: soil_type/cu/phi/N60/qu/RQD (+ optional rational-method
  fields N1_60/sigma_v_ref/OCR)
- SoilProfile adapter: to_drilled_shaft_input(shaft_length)
- LRFD phi factors: alpha=0.45, beta=0.55, rock=0.50 (redundant), etc.
- A cohesionless tip needs its N60 (GM-142, 2026-10-18): q_b = 57.5 N60
  kPa (N60 capped at 50) has no other input, so `require_tip_N60` refuses a
  tip layer with none, in `compute`, the calc steps and
  `capacity_vs_depth` (which used to leave those depths out). Before, N60 =
  15 was taken without saying so. On the side, a cohesionless layer with no
  N60 still gets no N60 < 15 reduction of beta (O'Neill & Reese).
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

## Side-resistance methods: simplified (default) vs rational (opt-in)

`DrillShaftAnalysis` selects the side-resistance method per soil type. Both
defaults reproduce the pre-v5.3 behavior byte-for-byte; the GEC-10 (2018)
*rational* chains are opt-in.

**Cohesionless — `beta_method`:**
- `"depth"` (default) — O'Neill & Reese (1999) depth-based
  `beta = 1.5 - 0.135*sqrt(z_ft) = 1.5 - 0.245*sqrt(z_m)` (unit-mixing defect FIXED 2026-07-19 — the old code applied 0.245 to feet, decaying ~1.81x too fast; found by NHI-06-089 Ex 9-5 curation), clamped to [0.25, 1.2] (with the N60<15
  reduction). `beta_cohesionless(z, N60)`.
- `"rational"` — GEC-10 Appendix A OCR/Ko chain (`beta_cohesionless_rational`):
  - `phi' = 27.5 + 9.2*log10[(N1)60]`   (`phi_prime_from_N1_60`, per-layer `N1_60`;
    falls back to `layer.phi` if `N1_60` unset)
  - `sigma'p = 0.47*pa*(N60)^0.6`   (`preconsolidation_stress`, Mayne 2007)
  - `OCR = sigma'p / sigma'v_ref`, where `sigma'v_ref` defaults to the current
    profile sigma'v but can be set per-layer to a **pre-scour** stress (GEC-10
    Appendix A uses the no-scour sigma'v for OCR while `fs` uses the scoured
    sigma'v). A direct per-layer `OCR` override bypasses the sigma'p/N60 step.
  - `Ko = (1 - sin phi')*OCR^(sin phi')`   (Mayne & Kulhawy 1982; capped at Kp)
  - `beta = Ko*tan(delta)`, `delta = phi'` (cast-in-place concrete against sand)

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

Both rational methods are validated against the GEC-10 Appendix A design example
(FHWA-NHI-18-024, Steps 11.5-11.6) in `validation_examples/test_published_v006_v008.py`:
Layer 3 sand beta = 0.41 (RSN 470.7 kips) and Layer 4 clay alpha = 0.47
(RSN 368.1 kips), reproduced through the high-level path to within ~1%.

## References (rational chains)
- FHWA GEC-10 (FHWA-NHI-18-024), 2018 — Appendix A design example; Section 10.3.5
- Chen et al. (2011) — alpha adhesion regression (Fig 10-6)
- Chen & Kulhawy (1993) — UU/UC -> CIUC strength transforms (Eq 10-16/10-17)
- Mayne & Kulhawy (1982) — Ko = (1 - sin phi')*OCR^(sin phi')
- Mayne (2007) — sigma'p correlation to N60
