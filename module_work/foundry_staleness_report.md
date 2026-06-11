# Foundry Staleness Report (v5.1 audit)

Date: 2026-06-10. Audit of `foundry/` (28 wrappers) and `foundry_test_harness/` against the
post-consolidation (v5.0) module roster and the v5.1 module changes committed in this worktree.

Validation environment: repo venv (`.venv\Scripts\python.exe`). The Palantir `functions.api`
SDK is NOT installed — wrappers use their built-in no-op `@function` fallback, so all
validation is offline behavior of the wrapped code, not the Foundry deploy path. That is a
missing-SDK condition, not staleness.

Validation performed:
- All 28 wrapper modules import clean.
- Full harness: **122/122 tests pass** (covers 16 wrappers, before and after fixes).
- All 13 `calc_package_agent` methods smoke-called end-to-end (HTML generated).
- `lateral_pile_agent` (not in harness) smoke-called: analysis + p-y curve OK.
- All 13 optional-dependency wrappers dispatch correctly (method registry path).
- Signature-level comparison against every v5.1-flagged module API.

## Prior interrupted attempt

The earlier run (killed by API error) had modified only `foundry/calc_package_agent_foundry.py`.
The diff was reviewed line-by-line against current module signatures and **kept — it is correct
and complete**: `analyze_slope` no longer takes `FOS_required` (moved to analysis dict);
Mononobe-Okabe `seismic_earth_pressure` result keys renamed to `*_kN_per_m`/`*_m`; the three
rewritten `ground_improvement` signatures (`analyze_aggregate_piers`, `analyze_surcharge_preloading`,
`analyze_vibro_compaction`) match the module exactly, including METHOD_INFO doc updates.
All touched paths smoke-verified OK.

## Per-wrapper status

| Wrapper | Module | Status | Drift found | Fixed |
|---|---|---|---|---|
| axial_pile_agent_foundry | axial_pile | CURRENT | None breaking. New optional `cohesive_phi`, `uplift_skin_fraction`, `pile_weight` not exposed (`include_uplift` is) | n (gap → v5.2) |
| bearing_capacity_agent_foundry | bearing_capacity | DRIFTED (doc) | `related` entry pointed at retired `dm7_agent` | y |
| calc_package_agent_foundry | calc_package (+12 modules) | DRIFTED → FIXED | `analyze_slope` FOS_required; MO result-key renames; 3 ground_improvement signature rewrites | y (prior attempt, verified) |
| downdrag_agent_foundry | downdrag | CURRENT | — | — |
| drilled_shaft_agent_foundry | drilled_shaft | CURRENT | v5.1 end-bearing cap is internal behavior; no stale wrapper claims | — |
| dxf_import_agent_foundry | dxf_import | CURRENT | — | — |
| groundhog_agent_foundry | external `groundhog` lib | CURRENT | Wraps the external PyPI groundhog directly (lazy imports), not a retired local module. Harness tier-3 green | — |
| ground_improvement_agent_foundry | ground_improvement | CURRENT | Harness green (wick/aggregate/vibro/feasibility) | — |
| gstools_agent_foundry | gstools_agent | CURRENT | — | — |
| hvsrpy_agent_foundry | hvsrpy_agent | CURRENT | hvsrpy lib not in venv; `has_hvsrpy` guard handles it | — |
| lateral_pile_agent_foundry | lateral_pile | CURRENT | Full smoke OK (banded solver transparent). New optional `stickup` not exposed | n (gap → v5.2) |
| liquepy_agent_foundry | liquepy_agent | CURRENT | No `spt_liquefaction` method (added to funhouse adapter in 5.0) | n (gap → v5.2) |
| opensees_agent_foundry | opensees_agent | CURRENT | — | — |
| pdf_import_agent_foundry | pdf_import | CURRENT | — | — |
| pile_group_agent_foundry | pile_group | CURRENT | Passes Mx/My straight through; v5.1 sign convention (+Mx uplifts +y) lives in the module; wrapper doc ("Mx causes variation in y-direction") is consistent | — |
| pystra_agent_foundry | pystra_agent | CURRENT | — | — |
| pystrata_agent_foundry | pystrata_agent | CURRENT | — | — |
| retaining_walls_agent_foundry | retaining_walls | CURRENT | `result.to_dict()` passthrough absorbs v5.1 thrust decomposition + MSE Meyerhof fields. New optional params (`include_passive`, `gamma_foundation`; MSE `phi_retained`/`gamma_retained`) not exposed | n (gap → v5.2) |
| salib_agent_foundry | salib_agent | CURRENT | — | — |
| seismic_geotech_agent_foundry | seismic_geotech | CURRENT | Already on the new API: `stress_reduction_rd(z)`, `compute_CSR(amax,sv,sve,z,M)`, `fines_correction(N160,FC)→N160cs`. The drift flagged in CONSOLIDATION_CHANGES.md (~line 385) is in `funhouse_agent/adapters/seismic_geotech.py`, NOT in foundry. `csr_crr_check` smoke-verified | — |
| seismic_signals_agent_foundry | seismic_signals_agent | CURRENT | — | — |
| settlement_agent_foundry | settlement | CURRENT | — | — |
| sheet_pile_agent_foundry | sheet_pile | DRIFTED (doc+param) | Doc claimed "Applies 1.2x design factor to computed embedment per USACE" — v5.1 removed the automatic 1.2x (FOS-basis change, `embedment_increase` default 1.0); param was not passable | y |
| slope_stability_agent_foundry | slope_stability | CURRENT | — | — |
| soe_agent_foundry | soe | CURRENT | — | — |
| subsurface_char_agent_foundry | subsurface_characterization | DRIFTED (doc) | `related`/`common_mistakes` referenced the retired standalone `pydiggs` surface; no methods yet for the consolidated `formats/` adapters (AGS4, GEF, DIGGS validation) | y (doc); format methods → v5.2 |
| swprocess_agent_foundry | swprocess_agent | CURRENT | — | — |
| wave_equation_agent_foundry | wave_equation | CURRENT | Already on `SoilSetup(R_ultimate=...)` and renamed BlowResult fields (`max_compression_stress`, `max_tension_stress`, `max_pile_force`). New optional `damping_model` not exposed | n (gap → v5.2) |

## Fixes applied this audit (mechanical only)

1. `foundry/calc_package_agent_foundry.py` — kept prior attempt's fix (verified correct).
2. `foundry/bearing_capacity_agent_foundry.py` — removed retired `dm7_agent` cross-reference.
3. `foundry/subsurface_char_agent_foundry.py` — pydiggs references updated to
   `subsurface_characterization.formats.diggs_validation.validate_diggs_schema`.
4. `foundry/sheet_pile_agent_foundry.py` — corrected the stale 1.2x-embedment doc to describe
   the v5.1 safety-basis choice (FOS_passive=1.5 default OR FOS=1.0 + embedment_increase
   1.2-1.4), and added the `embedment_increase` pass-through + param doc (default 1.0 = module
   default, behavior unchanged). Smoke-verified (D = converged x 1.3).

Post-fix: harness 122/122 green.

## Deletions

**None required.** No foundry wrappers exist for any retired module (geolysis, wind_loads,
pyseismosoil, fdm2d, DM7Eqs, Qt GUIs never had wrappers). The pygef/ags4/pydiggs surface was
already represented only by `subsurface_char_agent_foundry.py`, which targets the consolidated
module. `groundhog_agent_foundry.py` wraps the external groundhog library, not a local module.
No harness references to retired modules.

## Modules lacking a wrapper (report only)

- `dxf_export` — has a funhouse adapter; no foundry wrapper.
- `fem2d` — has a funhouse adapter; no foundry wrapper.
- (`geotech_common` is a shared library, `calc_package` is covered, `funhouse_agent` is the agent itself.)

## Harness assessment

`foundry_test_harness/` is fully offline-runnable (122 tests, ~27 s, repo venv). Covers 16 of
28 wrappers across 4 tiers (textbook values, workflows, cross-checks, error handling).
Not covered: calc_package, lateral_pile, dxf_import, pdf_import, and the 8 optional-dep
wrappers (gstools, hvsrpy, liquepy, opensees, pystrata, salib, seismic_signals, swprocess).
calc_package and lateral_pile were smoke-verified manually in this audit and are dependency-free —
they are the highest-value harness additions.

## Overall recommendation: KEEP, light refresh

The foundry surface survived the 5.0 consolidation and v5.1 API changes far better than the
funhouse adapters did: 23/28 wrappers fully current, 5 had only doc-level or already-fixed
drift, zero broken imports, zero retired-module wrappers. The harness is healthy and fast.
Retiring is not warranted; a light v5.2 refresh closes the remaining gaps.

## Proposed v5.2 actions

1. **Expose new optional params** (one-line pass-throughs + METHOD_INFO docs):
   - wave_equation: `damping_model` ('smith'/'case' etc.)
   - lateral_pile: `stickup`
   - axial_pile: `cohesive_phi`, `uplift_skin_fraction`, `pile_weight`
   - retaining_walls: `include_passive`, `gamma_foundation` (cantilever); `phi_retained`, `gamma_retained` (MSE)
2. **subsurface_char format methods**: add `read_ags4`/`validate_ags4`, `parse_cpt_file`/`parse_bore_file`
   (GEF), `validate_diggs_schema` wrappers for the consolidated `formats/` subpackage.
3. **liquepy parity**: add `spt_liquefaction` method mirroring the funhouse adapter.
4. **Decide dxf_export / fem2d**: add wrappers or document as intentionally out of foundry scope.
5. **Harness coverage**: add tier-1 tests for `calc_package_agent` (all 13 methods are
   dependency-free) and `lateral_pile_agent`.
6. **Drift prevention**: consider generating the foundry method registries from
   `funhouse_agent/adapters/MODULE_REGISTRY` (single source of truth) so the two consumer
   surfaces cannot diverge again; alternatively add a CI check that imports foundry wrappers
   and asserts wrapped-call signatures via `inspect.signature` binding.
7. **Cleanup**: delete stale `foundry/__pycache__` artifacts from the repo if they are tracked
   (they appear to be build artifacts only).
