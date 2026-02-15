# wave_equation — Smith 1-D Wave Equation Analysis

## Purpose
Simulates pile driving using Smith (1960) 1-D wave equation method.
Produces bearing graphs (capacity vs blow count) and drivability analyses.

## References
- Smith (1960) — original wave equation formulation
- Goble & Rausche (1976) — WEAP methodology
- FHWA GEC-12 (Driven Piles) — wave equation chapter

## Files
- `hammer.py` — 14 built-in hammers (diesel, hydraulic, air/steam)
- `cushion.py` — hammer cushion and pile cushion stiffness
- `pile_model.py` — mass-spring discretization of pile
- `soil_model.py` — Smith quake and damping parameters
- `time_integration.py` — explicit central difference time stepping
- `bearing_graph.py` — capacity vs blow count analysis
- `drivability.py` — blow count vs depth profile
- `results.py` — WaveEqResult, BearingGraphResult, DrivabilityResult

## Public API
```python
analyze_bearing_graph(hammer, pile, soil, capacities, ...) -> BearingGraphResult
analyze_drivability(hammer, pile, soil_profile, ...) -> DrivabilityResult
```

## Key Notes
- Time step automatically chosen for Courant stability
- Smith damping: R = R_static * (1 + J*v) where J = damping factor
- Quake: elastic limit of soil spring (typically 2.5mm shaft, 2.5-5mm toe)
- 41 tests
