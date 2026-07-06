# sheet_pile — Sheet Pile Wall Design

## Purpose
Analyzes cantilever and anchored sheet pile walls using Rankine or Coulomb
earth pressure theory. Computes embedment depth, maximum moment, and
anchor force.

## References
- Rankine (1857) — active/passive pressure coefficients
- Coulomb (1776) — with wall friction
- USS Steel Sheet Piling Design Manual

## Files
- `earth_pressure.py` — Ka, Kp, active/passive pressure distributions
- `cantilever.py` — cantilever wall embedment + moment
- `anchored.py` — free-earth support method
- `results.py` — SheetPileResult with summary()/to_dict()

## Public API
```python
analyze_sheet_pile(wall_type, excavation_depth, layers, ...) -> SheetPileResult
```

## Key Notes
- earth_pressure module reused by retaining_walls/earth_pressure.py
- Layers are dicts with phi, c, gamma, thickness

## Cantilever safety basis (SP-3, changed in v5.1)
The classic simplified cantilever method uses ONE of two safety bases:
1. **FS on passive resistance** (~1.5-2.0) with NO embedment increase, or
2. **Unfactored passive (FS ≈ 1.0)** with a 20-40% embedment increase.

Prior to v5.1 `analyze_cantilever` applied both (FOS_passive = 1.5 AND a
hardcoded 1.2× depth increase) — a double safety margin. Since v5.1 the
default is basis 1: `FOS_passive = 1.5`, `embedment_increase = 1.0` (a new,
documented parameter). Pass `FOS_passive = 1.0, embedment_increase = 1.2-1.4`
for basis 2. Result records both `embedment_converged` and
`embedment_increase`; calc_steps renders whichever basis was used.
**Behavior change:** default design embedment is 1/1.2 of the pre-v5.1 value
(the old default combination was conservative double-counting); max moment is
computed over the (shorter) design embedment accordingly.
Anchored walls (free earth support) are unchanged: factored passive, no
depth increase.

## Log-spiral passive coefficient (SP-4, v5.3)
`earth_pressure.caquot_kerisel_Kp(phi, delta, Kp_initial=None)` adds the
Caquot-Kerisel (1948) log-spiral passive coefficient: Kp' = R·Kp0, where Kp0 is
the passive coefficient at δ=φ (digitized from the C-K chart; the φ=30 anchor is
the Caltrans Fig 4-20 value 6.30) and R = Kp(δ)/Kp(δ=φ) ≤ 1 is the wall-friction
reduction (Caltrans T&S Matrix 4-1 / NAVFAC DM-7.2, tabulated φ 30-35, δ/φ
0.40-0.50). Unlike Coulomb (planar wedge), the log spiral does NOT over-predict
Kp at high δ/φ. For φ=30, δ/φ=0.5, Kp' = 6.30·0.746 = 4.70 (Caltrans Ex 8-1).
Selectable via `pressure_method="log_spiral"` in `analyze_cantilever` /
`analyze_anchored` (uses Coulomb active + the log-spiral passive; set the layer
`wall_friction_deg` = δ). Default remains Rankine/Coulomb. Validated in
`validation_examples` (V-013).
