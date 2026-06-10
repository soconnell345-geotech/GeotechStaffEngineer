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
- Cantilever checks decompose the active thrust per the chosen method (RW-1):
  Coulomb thrust acts at δ = 2/3·φ from horizontal, sloped-Rankine thrust
  parallel to the slope (β). Only P_h = Pa·cos(δ) drives sliding/overturning;
  P_v = Pa·sin(δ) is applied downward at the back of the heel (x = B) and is
  stabilizing. Rankine + level backfill is unchanged (δ = 0).
- Sloped backfill adds the triangular soil wedge above the heel to the
  weight/moment tally (RW-4)
- MSE internal: Kr/Ka stress ratio varies with depth (AASHTO method).
  Active-zone length La branches on reinforcement extensibility (RW-2):
  bilinear coherent-gravity surface (0.3H) for metallic/inextensible,
  Rankine 45+φ/2 plane for geosynthetic/extensible (AASHTO Fig. 11.10.6.3.1-1)
- MSE external bearing uses the Meyerhof uniform pressure over the effective
  width, σ_v = W/(L−2e), per AASHTO 11.10.5.4 (RW-3) — not trapezoidal q_toe
- SoilProfile adapter: to_retaining_wall_input(wall_height, surcharge)
