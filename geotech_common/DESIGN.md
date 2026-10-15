# geotech_common — Shared Utilities

## Purpose
Three components shared across all modules:
1. **SoilProfile** — unified soil model with adapter methods to all modules
2. **Engineering Checks** — 9 review functions returning List[str] warnings
3. **Correlations & Utilities** — unit conversions, soil parameter correlations

## Files
- `soil_profile.py` — SoilLayer (30+ fields), GroundwaterCondition, SoilProfile, SoilProfileBuilder
- `engineering_checks.py` — 9 check functions (bearing, settlement, pile, lateral, sheet, wave_eq, group, selection, consistency)
- `tables.py` — `CalcTable`, the calculation tables a result returns from `get_tables()` (GM-39, 2026-10-14; written as CSVs and printed by `calc_package`)
- `tests/test_soil_profile.py` — 82 tests (9 test classes)
- `tests/test_engineering_checks.py` — 93 tests
- `tests/test_adapters.py` — 84 tests (incl. integration round-trips)
- `tests/test_tables.py` — 29 tests; `test_plot_methods.py` 24, `test_soil_properties.py` 16

## SoilProfile Adapters (11 methods)
Each returns plain dicts (no circular imports):
- `to_bearing_capacity_input(footing_depth)`
- `to_settlement_input(footing_depth, B, L)`
- `to_axial_pile_input(pile_length)`
- `to_lateral_pile_input(pile_length, pile_diameter, loading)`
- `to_sheet_pile_input(excavation_depth)`
- `to_soe_input(excavation_depth, support_depths, support_type, support_spacing, surcharge)`
- `to_pile_group_input(pile_length, pile_diameter)`
- `to_drilled_shaft_input(shaft_length)`
- `to_retaining_wall_input(wall_height, surcharge)`
- `to_seismic_input(amax_g, magnitude)`
- `to_slope_stability_input(ground_elevation)` — depth-based layers to elevation-based `SlopeSoilLayer` dicts

## Key Notes
- SoilProfile is depth-based (top=0, depth increases down)
- slope_stability uses elevation-based SlopeSoilLayer (separate; `to_slope_stability_input` converts)
- fill_missing_from_correlations() tracks estimated vs measured via _estimated_fields
- Correlations: N60->phi (Peck), N60->cu (Terzaghi-Peck), LL->Cc, Cc->Cr, cu->eps50
- SoilProfileBuilder: from SPT boring log, CPT data, or simple dict table
- 328 tests in all (collected 2026-10-15)

## Correlation provenance
The empirical correlations in `soil_properties.py` are **approximate,
for-preliminary-design** estimates — each carries an inline source citation, but
NONE is a primary-source-in-hand transcription and several are hand-digitized
chart fits with author-chosen interpolation. Use measured values for final
design. Source basis per correlation:
- **N60 -> phi (Peck)** — hand-digitization of the Peck, Hanson & Thornburn
  (1974) N-vs-phi chart; breakpoints trace the chart zones but intermediate
  slopes are author-chosen and NOT verified per-value against PHT 1974
  Table 10-3. (Also `meyerhof` phi ~ 25 + 0.3*N60, an approximate closed form.)
- **N60 -> cu (Terzaghi-Peck), LL -> Cc, Cc -> Cr, cu -> eps50** — standard
  textbook correlations coded from the formula; approximate.
- **GAMMA_W (water.py)** — physical constant (9.81 kN/m^3), not a correlation.

Wiki-wishlist: Peck, Hanson & Thornburn (1974) Table 10-3 / their chart, to
re-verify the N->phi breakpoints and slopes. See
`module_work/provenance_audit_other.md` for the full audit.
