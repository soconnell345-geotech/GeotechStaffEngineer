# bearing_capacity — Shallow Foundation Bearing Capacity

## Purpose
Computes ultimate and allowable bearing capacity for shallow footings using
Vesic (1973) and Meyerhof (1963) general bearing capacity equations, matching
CBEAR program output.

## References
- Vesic (1973) — general bearing capacity factors Nc, Nq, Ngamma
- Meyerhof (1963) — shape/depth/inclination factors
- FHWA GEC-6 (Shallow Foundations)

## Files
- `capacity.py` — `vesic_bearing_capacity()`, `meyerhof_bearing_capacity()`
- `factors.py` — Nc, Nq, Ngamma, shape/depth/inclination factor functions
- `results.py` — BearingCapacityResult with summary()/to_dict()
- `__init__.py` — exports analyze_bearing_capacity()

## Public API
```python
analyze_bearing_capacity(B, L, D, phi, c, gamma, ...) -> BearingCapacityResult
```

## Key Notes
- Two-layer support: layer1 (footing) + layer2 (below) with weighted averaging
- GWT depth adjusts effective unit weight below water table
- 37 tests in tests/test_bearing_capacity.py
