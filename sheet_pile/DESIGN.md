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
- 26 tests
