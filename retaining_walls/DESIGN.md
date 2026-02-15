# retaining_walls — Cantilever & MSE Wall Design

## Purpose
Stability analysis for cast-in-place cantilever retaining walls and
mechanically stabilized earth (MSE) walls per GEC-11.

## References
- FHWA GEC-11 (Mechanically Stabilized Earth Walls)
- AASHTO LRFD Bridge Design Specifications
- Coulomb/Rankine earth pressure theory

## Files
- `geometry.py` — CantileverWallGeometry (auto-sizing), MSEWallGeometry
- `earth_pressure.py` — reuses sheet_pile Ka/Kp + resultant force computation
- `cantilever.py` — sliding, overturning, bearing, eccentricity checks
- `mse.py` — Kr/Ka ratio, Tmax, pullout, F*, external + internal stability
- `reinforcement.py` — Reinforcement dataclass + built-in products
- `results.py` — CantileverWallResult, MSEWallResult

## Public API
```python
analyze_cantilever_wall(geometry, soil_params, ...) -> CantileverWallResult
analyze_mse_wall(geometry, reinforcement, soil_params, ...) -> MSEWallResult
list_reinforcement() -> list of built-in reinforcement products
```

## Key Notes
- CantileverWallGeometry auto-sizes stem thickness, toe/heel from wall height
- MSE internal: Kr/Ka stress ratio varies with depth (AASHTO method)
- SoilProfile adapter: to_retaining_wall_input(wall_height, surcharge)
- 46 tests
