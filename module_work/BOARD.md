# Module-Improvement BOARD

Top-level backlog and status. Maintained by the team lead.

Status: ☐ todo · ◐ in progress · ☑ done · ⚠ blocked

## Baseline metric (funhouse native tool-calling suite)

`funhouse_agent/geotech_test_suite.json`, 68 Qs / 41 modules, model = 5.1 (2026-06-03):
**clean 28 · recovered-with-errors 40 · failed 0.** Goal: drive "recovered" down by
fixing tool ergonomics + real per-module issues. Re-run after each batch to measure.

## Domain roster

| Domain | Ledger | Modules |
|--------|--------|---------|
| foundations | `foundations.md` | bearing_capacity, settlement, retaining_walls, ground_improvement, downdrag |
| deep-foundations | `deep-foundations.md` | axial_pile, drilled_shaft, lateral_pile, pile_group, wave_equation |
| earth-retention | `earth-retention.md` | sheet_pile, soe |
| slope-fem | `slope-fem.md` | slope_stability, fem2d |
| seismic | `seismic.md` | seismic_geotech, pystrata_agent, opensees_agent, liquepy_agent, seismic_signals_agent |
| characterization | `characterization.md` | subsurface_characterization (now also holds the folded GEF/AGS4/DIGGS-validation format adapters), hvsrpy_agent, swprocess_agent, gstools_agent, salib_agent, pystra_agent |
| io-cad | `io-cad.md` | dxf_import, dxf_export, pdf_import |
| references | `references.md` | dm7, gec6/7/10/11/12/13, micropile, fema_p2192, noaa_frost, ufc_* , reference_db |
| common (lead-serialized) | — | geotech_common |

## Backlog

### Phase 0 — shared ergonomics fix (lead) — ☑ DONE 2026-06-03
- ☑ `allowed_values` convention added to METHOD_INFO enums in the 3 pilot adapters
  (axial_pile, bearing_capacity, slope_stability); soil_type / width-vs-diameter /
  analysis_mode descriptions clarified. `describe_method` surfaces it automatically.
- ☑ Fixed native-system-prompt bug in `funhouse_agent/agent.py` — regex sentinel was
  `## Module Catalog` (never existed) so the native prompt was dropping the whole
  module catalog, forcing an extra `list_agents` round. Now keyed to
  `## Available Modules`.
- ☑ Added `## Tool Discipline` nudge to `funhouse_agent/system_prompt.py` (placed
  before `## ReAct Protocol` so it survives the native strip): describe_method first,
  use only allowed_values, don't invent method names, prefer search_critical_surface.
- ☑ Verified: `funhouse_agent/tests/` green except one unrelated pre-existing failure
  (see Known issues). Native prompt now retains catalog + nudge.

### Rollout — extend the convention (per domain, ☐ todo)
- ☐ Add `allowed_values` to METHOD_INFO enum params in the remaining ~47 adapters,
  domain by domain (each specialist does its own `funhouse_agent/adapters/<module>.py`).

### Triage — funhouse feedback (lead, ⚠ blocked on upload)
- ⚠ Parse `geotech_test_suite_results.json` (awaiting user upload) into per-domain
  tasks below. The 40 recovered-with-errors records carry the specifics.

### Known issues (logged from test runs)
- ☐ **references**: `ufc_pavement` adapter `test_method_count` fails (11 actual vs 9
  expected in `funhouse_agent/tests/test_reference_adapters.py`). Either 2 methods were
  added without updating the count, or the adapter exposes extras. References specialist
  to reconcile.

## Triage results — baseline 4.6.1 (2026-06-03)

Source: `docs/geotech_test_suite_results.json` → full detail in `module_work/module_feedback.json`
(regenerate via `python module_work/triage_feedback.py`). **NOTE: this baseline predates the
Phase 0 fixes** (ran on published 4.6.1), so the guessing categories below should shrink sharply
once Phase 0 ships. Error categories across the 40 recovered records:

1. **method-name guessing (~38 hits, ~22 modules)** — model invents method names
   (`terzaghi_general`, `beta_method`, `rock_socket_capacity`, `basal_heave_fos`,
   `strip_footing_srm`, `equivalent_linear`, …). **Cross-cutting, NOT per-module.** Addressed by
   Phase 0 (native-catalog fix + Tool Discipline nudge). Measure after re-publish.
2. **module-name guessing (~8 hits)** — `earth_pressure`, `consolidation`, `deep_foundation`,
   `stress_distribution`, … Also cross-cutting; the restored module catalog should largely fix it.
3. **param-name mismatches (~14 hits, ~9 modules)** — REAL per-module adapter bugs: METHOD_INFO
   documents/accepts a param the underlying function doesn't take, or accesses `params["x"]` with a
   raw KeyError. The genuine specialist work. Affected: axial_pile (width/diameter/wall_thickness),
   ground_improvement (area_replacement_ratio, drain_spacing), retaining_walls (phi→rankine/coulomb),
   sheet_pile (unit_weight), soe (thickness), lateral_pile (k),
   slope_stability (name), reference_db (max_results).
4. **missing optional deps in Databricks (~6 modules)** — NOT code bugs: liquepy, pystrata,
   opensees(py), eqsig (seismic_signals), SALib, pystra are not installed in the
   Databricks runtime. Fix = `pip install geotech-staff-engineer[full]` (or specific extras) there.
5. **enum values (~4)** — downdrag `soil_type` (`clay`/`settling_fill`), axial_pile variants.
   Phase 0 `allowed_values` rollout (per domain).
6. **slope geometry (3)** — slope_stability guessed trial circles that miss the surface. Phase 0
   steer to `search_critical_surface` should resolve.

**Recommended sequencing:** (a) commit + publish Phase 0 as 4.6.2, re-run the suite in Databricks,
re-triage → measures how much #1/#2/#5/#6 clear for free; (b) specialists then tackle the ~9 modules
with real param-name bugs (#3); (c) install extras in Databricks for #4.

## Pilot (Phase 2, after triage)
Three specialists, three domains: deep-foundations (axial_pile), foundations
(bearing_capacity), slope-fem (slope_stability). Worktree-isolated, test-gated,
diffs reviewed by lead before merge.
