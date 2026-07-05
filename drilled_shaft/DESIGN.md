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

## Side-resistance methods: simplified (default) vs rational (opt-in)

`DrillShaftAnalysis` selects the side-resistance method per soil type. Both
defaults reproduce the pre-v5.3 behavior byte-for-byte; the GEC-10 (2018)
*rational* chains are opt-in.

**Cohesionless — `beta_method`:**
- `"depth"` (default) — O'Neill & Reese (1999) depth-based
  `beta = 1.5 - 0.245*sqrt(z_ft)`, clamped to [0.25, 1.2] (with the N60<15
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
