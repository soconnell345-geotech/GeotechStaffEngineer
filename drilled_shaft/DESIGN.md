# drilled_shaft — Drilled Shaft Axial Capacity

## Purpose
Computes axial capacity of drilled shafts (bored piles) using GEC-10 methods:
alpha method (clay), beta method (sand), and rock socket design.

## References
- FHWA GEC-10 (Drilled Shafts)
- O'Neill & Reese (1999) — alpha method for clay
- AASHTO LRFD Bridge Design Specifications — resistance factors

## Files
- `shaft.py` — DrillShaft geometry (diameter, socket, bell)
- `soil_profile.py` — ShaftSoilLayer and ShaftSoilProfile
- `side_resistance.py` — alpha_method(), beta_method(), rock_socket()
- `end_bearing.py` — clay_Nc(), sand_N60(), rock_UCS()
- `capacity.py` — DrillShaftAnalysis orchestrator
- `lrfd.py` — AASHTO phi factors by method and condition
- `results.py` — DrillShaftResult with summary()/to_dict()

## Public API
```python
analyze_drilled_shaft(shaft, soil_profile, ...) -> DrillShaftResult
analyze_capacity_vs_depth(shaft, soil_profile, depths, ...) -> list
get_resistance_factors(method, condition) -> dict
```

## Key Notes
- ShaftSoilLayer dicts: soil_type/cu/phi/N60/qu/RQD
- SoilProfile adapter: to_drilled_shaft_input(shaft_length)
- LRFD phi factors: alpha=0.45, beta=0.55, rock=0.50 (redundant), etc.
- 41 tests
