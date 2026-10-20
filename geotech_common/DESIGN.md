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
- `choices.py` — what a module chose, as a result's `choices` list (GM-115, 2026-10-16): one entry per choice, `{what, chosen, how: given | default | module, basis, alternatives}` (`basis` and `alternatives` left out when empty, to keep tool results short), built with `choice()` and `how_given()`; the text is plain words with any input or result key the agent needs in brackets ("Boussinesq (stress_method 'boussinesq')"), which `plain()` takes out for `summary()` (`choices_lines()`) and the calc package's "Method Choices" table (`choice_rows()`). Citations are checked against the library's PDFs (GEC-12 as printed, FHWA-NHI-16-009). Filled first in axial_pile (what `method="auto"` resolved to by soil type), lateral_pile (each layer's p-y model and its source), settlement (the stress increase, Iw, the drainage path, t1), seismic_geotech (M-O, kh/kv, delta, the static KA, the height of delta-PAE; liquefaction's procedure, K_sigma left out; rd and MSF per layer), liquepy_agent (B&I 2014 SPT/CPT; rd, MSF, K_sigma per SPT layer) and slope_stability's search (the families scored, with why one it ran scored none, and those not tried; the ranges, pointing at `search_ranges`). The agent layer adds `defaults_used` beside them (`funhouse_agent/dispatch.py`). `call_inputs()` (GM-118, 2026-10-17) tells `how_given(value, default, name=)` which inputs the call named and which declared defaults it left in force, so a typed value equal to the default is "given"; outside a call, and for a value nested in an input, `how_given` compares by value; every name it calls "default" is in `defaults_used`
- `log_spiral.py` — the log-spiral passive Kp as the references print it (GM-183, 2026-10-19): the Caquot-Kerisel (1948) chart with its R table (Caltrans T&S Fig. 4-20, level and sloping ground), UFC 3-220-20 Fig. 4-13 (Kerisel & Absi 1990, δ/φ′ = 0.66) and Eq. 4-12; chart values traced off the rendered pages; shared by `soe` and `sheet_pile` (`caquot_kerisel_Kp`) and `earth_pressure_coefficient`
- `plan_furniture.py` — the scale bar (four black and white segments, a round 1, 2 or 5 × 10^k length, named at its ends) and north arrow (`north_deg` clockwise from up; straight up with a note when not given) of a plan drawn as a drawing, in matplotlib and Plotly; shared by `profile_figure.plot_data(plan_view=True)` and the `gstools` probability map so the plans read alike (2026-10-15)
- `layout_ink.py` — the layout check by drawn pixels (GM-108): lettering that overlaps or runs into other lettering, that a line, an edge, markers or a leader arrow shows through, or that an edge cuts, measured by drawing each leaf of a figure alone as it is saved; shared by the plotting box (`funhouse_agent/plot_box/child.py`, which loads it by its file path) and `plot_data`'s notes (`layout_check(..., texts=)`), GM-109. The check swaps `draw` on matplotlib's base classes while it saves, so one check runs at a time (`_LOCK`) and a draw on any other thread is the original draw. Stdlib only at the top (2026-10-18)
- `tests/test_soil_profile.py` — 82 tests (9 test classes)
- `tests/test_engineering_checks.py` — 93 tests
- `tests/test_adapters.py` — 84 tests (incl. integration round-trips)
- `tests/test_tables.py` — 29 tests; `test_plot_methods.py` 24, `test_soil_properties.py` 16, `test_plan_furniture.py` 3, `test_layout_ink.py` 9

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
- 331 tests in all (collected 2026-10-15)

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
