# geotech_common — Shared Utilities

## Purpose
Three components shared across all modules:
1. **SoilProfile** — unified soil model with adapter methods to all modules
2. **Engineering Checks** — 9 review functions returning List[str] warnings
3. **Correlations & Utilities** — unit conversions, soil parameter correlations

## Files
- `soil_profile.py` — SoilLayer (30+ fields), GroundwaterCondition, SoilProfile, SoilProfileBuilder
- `engineering_checks.py` — 9 check functions (bearing, settlement, pile, lateral, sheet, wave_eq, group, selection, consistency)
- `tests/test_soil_profile.py` — 82 tests (9 test classes)
- `tests/test_engineering_checks.py` — 84 tests (9 test classes)
- `tests/test_adapters.py` — 72 tests (11 test classes, incl. integration round-trips)

## SoilProfile Adapters (9 methods)
Each returns plain dicts (no circular imports):
- `to_bearing_capacity_input(footing_depth)`
- `to_settlement_input(footing_depth, B, L)`
- `to_axial_pile_input(pile_length)`
- `to_lateral_pile_input(pile_length, pile_diameter, loading)`
- `to_sheet_pile_input(excavation_depth)`
- `to_pile_group_input(pile_length, pile_diameter)`
- `to_drilled_shaft_input(shaft_length)`
- `to_retaining_wall_input(wall_height, surcharge)`
- `to_seismic_input(amax_g, magnitude)`

## Key Notes
- SoilProfile is depth-based (top=0, depth increases down)
- slope_stability uses elevation-based SlopeSoilLayer (separate, no adapter yet)
- fill_missing_from_correlations() tracks estimated vs measured via _estimated_fields
- Correlations: N60->phi (Peck), N60->cu (Terzaghi-Peck), LL->Cc, Cc->Cr, cu->eps50
- SoilProfileBuilder: from SPT boring log, CPT data, or simple dict table
- 238 total tests (82 + 84 + 72)
