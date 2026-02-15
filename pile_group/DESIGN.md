# pile_group — Rigid Cap Pile Group Analysis

## Purpose
Analyzes pile groups under combined loading (6-DOF) with rigid cap
assumption. Includes group efficiency and p-multiplier corrections.

## References
- Converse-Labarre formula for group efficiency
- FHWA GEC-12 (Driven Piles) — group effects
- Reese & Van Impe (2001) — p-multipliers

## Files
- `pile_layout.py` — GroupPile dataclass, rectangular_layout() generator
- `group_efficiency.py` — converse_labarre(), block_failure(), p_multiplier()
- `rigid_cap.py` — simplified elastic + 6-DOF stiffness matrix
- `results.py` — PileGroupResult with summary()/to_dict()

## Public API
```python
analyze_pile_group(piles, loads, soil_params, ...) -> PileGroupResult
```

## Key Notes
- Piles defined by (x, y) positions in plan view
- 6-DOF: Vx, Vy, Vz (axial), Mx, My, Mz (torsion)
- SoilProfile adapter: to_pile_group_input() returns weighted avg phi/cu
- 22 tests
