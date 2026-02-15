# axial_pile — Driven Pile Axial Capacity

## Purpose
Computes ultimate axial capacity (skin friction + end bearing) for driven
piles using Nordlund, Tomlinson alpha, and effective stress (beta) methods.

## References
- Nordlund (1963, 1979) — sand skin friction with delta/phi correction
- Tomlinson (1957) — alpha method for clay
- Effective stress (beta) method — Burland (1973)
- FHWA GEC-12 (Driven Piles)

## Files
- `nordlund.py` — nordlund_skin_friction(), nordlund_end_bearing()
- `tomlinson.py` — tomlinson_skin_friction()
- `beta_method.py` — beta_skin_friction()
- `capacity.py` — analyze_axial_pile() orchestrator
- `results.py` — AxialPileResult with summary()/to_dict()

## Public API
```python
analyze_axial_pile(pile_length, pile_diameter, layers, method, ...) -> AxialPileResult
```

## Key Notes
- Layers are dicts with soil_type ("sand"/"clay"), phi/cu, gamma, thickness
- SoilProfile adapter: to_axial_pile_input(pile_length) clips layers to pile length
- 36 tests
