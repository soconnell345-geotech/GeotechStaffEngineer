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

## Known Simplifications (read before relying on Nordlund numbers)
- **Nordlund chart fits** (`nordlund.py`): `nordlund_Kd` ignores the
  displaced-volume (V/V0) curve family (fixes the displacement-pile curve);
  `alpha_t_factor` ignores phi and saturates at 1.0 for D/b > 5 (tip can be
  over-predicted, bounded by the Meyerhof q_L cap); `nordlund_CF` is a linear
  fit keyed only on delta/phi. Results can deviate from a rigorous
  chart-based Nordlund calculation — verify against the GEC-12 charts or a
  load test where tip resistance or low-displacement piles govern. (No
  numeric GEC-12 worked example is available in the geotech-references text
  layer to pin the deviation; flagged for a future tie-out.)
- **delta/phi defaults**: 0.75 steel, 0.90 concrete/timber
  (`nordlund.delta_from_phi`, GEC-12 Table 7-1) — overridable per layer via
  `AxialSoilLayer.delta_phi_ratio`.
- **Beta method clay phi**: cohesive layers assume phi' = 25 deg by default;
  overridable via `AxialPileAnalysis(cohesive_phi=...)`.
- **Skin friction integration**: midpoint rule per layer, with segments
  split at the GWT (exact for the piecewise-linear sigma_v' profile).
- **Uplift**: rule-of-thumb `uplift_skin_fraction` (default 0.75) x OUTSIDE
  skin friction only; inside (plug) friction excluded; pile self-weight only
  if supplied via `pile_weight`. Not a substitute for a dedicated tension
  design method.
