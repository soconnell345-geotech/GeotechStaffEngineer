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
- `log_spiral.py` — the log-spiral passive Kp as the references print it (GM-183, 2026-10-19): the Caquot-Kerisel (1948) chart with its R table (Caltrans T&S Fig. 4-20, level and sloping ground), UFC 3-220-20 Figs. 4-12 and 4-13 (Kerisel & Absi 1990, δ/φ′ = 0.66) and Eq. 4-12; chart values traced off the rendered pages; shared by `soe` and `sheet_pile` (`caquot_kerisel_Kp`) and `earth_pressure_coefficient`. Since GM-232 (2026-10-20, owner OWN-66) the default source is `ufc_3_220_20`, UFC's source by the wall's geometry (pp. 191-193): Eq. 4-12 for θ = 0 and β = 0 at any δ, Fig. 4-12 for a battered face (θ −30 to +40°) with level ground, Fig. 4-13 for a vertical face with sloping ground, both charts at δ/φ′ = 0.66 only; a case UFC does not print (θ and β together, either chart at another δ) is refused with the route that covers it named (the 1948 chart, vertical wall, any δ). θ is UFC's face angle from vertical, positive when the soil overhangs the face (θ = 90° − Coulomb's α)
- `alpha_method.py` — the alpha method's charts for driven piles in clay as the references print them (GM-255, 2026-10-20): UFC 3-220-20 Table 6-16 (soft clay below 800 psf, α 1), Fig 6-14 curves A (Z ≥ 40b) and B (Z = 10b) read off the figure's vector paths (`alpha_ufc`, linear in Z/b between, held at curve B below 10b and at the curves' end past su/Pa 2.357, each said in `notes`), Eq 6-9 for the transition zone (`alpha_ufc_eq_6_9`, 0.28 (su/Pa)^-0.44) and `ufc_consistency`; and FHWA GEC-12 Fig. 7-17 as drawn (`adhesion_fig_7_17`, measured 2026-10-18 for `axial_pile`, GM-229, moved here so `downdrag` reads the same curves; `axial_pile.tomlinson` re-exports it). `downdrag` reads UFC's by default and GEC-12's as its option
- `plan_furniture.py` — the scale bar (four black and white segments, a round 1, 2 or 5 × 10^k length, named at its ends) and north arrow (`north_deg` clockwise from up; straight up with a note when not given) of a plan drawn as a drawing, in matplotlib and Plotly; shared by `profile_figure.plot_data(plan_view=True)` and the `gstools` probability map so the plans read alike (2026-10-15)
- `layout_ink.py` — the layout check by drawn pixels (GM-108): lettering that overlaps or runs into other lettering, that a line, an edge, markers or a leader arrow shows through, or that an edge cuts, measured by drawing each leaf of a figure alone as it is saved; shared by the plotting box (`funhouse_agent/plot_box/child.py`, which loads it by its file path) and `plot_data`'s notes (`layout_check(..., texts=)`), GM-109. The check swaps `draw` on matplotlib's base classes while it saves, so one check runs at a time (`_LOCK`) and a draw on any other thread is the original draw. Stdlib only at the top (2026-10-18)
- `tests/test_soil_profile.py` — 82 tests (9 test classes)
- `tests/test_engineering_checks.py` — 93 tests
- `tests/test_adapters.py` — 84 tests (incl. integration round-trips)
- `tests/test_tables.py` — 29 tests; `test_plot_methods.py` 24, `test_soil_properties.py` 16, `test_plan_furniture.py` 3, `test_layout_ink.py` 9, `test_alpha_method.py` 8

## SoilProfile Adapters (11 methods)
Each returns plain dicts (no circular imports):
- `to_bearing_capacity_input(footing_depth)`
- `to_settlement_input(footing_depth, B, L)`
- `to_axial_pile_input(pile_length)`
- `to_lateral_pile_input(pile_length, pile_diameter, loading, n_cycles, free_water_at_surface, stiff_clay_cu)` — returns `layers` and `choices`. Stiff clay (cu `stiff_clay_cu` and up) takes Reese et al. (1975) only with free water at the ground surface, Welch & Reese otherwise, below the water table too (GEC 9 p. 219; `free_water_at_surface` overrides; GM-215); `n_cycles` (N) reaches every cyclic Welch & Reese layer, which fails without it (GM-209); an estimated eps50 or k is listed with its table (COM624P Tables 3.2, 3.3, edges at 0.5/1/2 ton/ft^2)
  - **The soft/stiff clay boundary (GM-217)**, `stiff_clay_cu`: a clay is stiff from 100 kPa by default, this module's own value (`STIFF_CLAY_CU`). GEC 9 prints the clays each criterion was developed on, soft clays (Matlock 1970) 250-500 psf (12.0-23.9 kPa, p. 216) and stiff clays (Reese et al. 1975) 1,000-2,000 psf (47.9-95.8 kPa, p. 219), and no boundary between them; COM624P prints none either (its stiff clay Tables 3.2 and 3.3 start at 0.5 ton/ft^2, 47.9 kPa). The default sits above GEC 9's stiff range, so a 48-100 kPa clay, stiff in GEC 9's words, takes Matlock's soft clay; which default to use is the owner's call. The result's `choices` state the boundary (given or default), the printed ranges and what 47.9 kPa would change in the profile. A `free_water_at_surface` given with no stiff clay below the water table is listed as "not used". No number changes.
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
