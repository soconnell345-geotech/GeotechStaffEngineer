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
