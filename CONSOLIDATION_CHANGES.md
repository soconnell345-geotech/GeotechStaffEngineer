# Consolidation Phase 1 — Changes Manifest

Authoritative record of Phase 1 of the module-consolidation effort (see
`CONSOLIDATION_PLAN.md`). Executed on branch `scope-consolidation` in the
`scope-consol` worktree, 2026-06-09.

## Modules removed (4)

These 4 redundant/out-of-scope analysis modules were deleted in full (module
dir + adapter + foundry wrapper + tests + all wiring). None had importers among
the kept analysis modules.

| Module | Reason | Kept alternative |
|--------|--------|------------------|
| `pyseismosoil_agent` | Redundant site-response/soil-curve wrapper | `pystrata_agent` + `opensees_agent` |
| `fdm2d` | Redundant continuum solver | `fem2d` |
| `geolysis_agent` | Duplicates native bearing_capacity + groundhog | `bearing_capacity` + `groundhog_agent` |
| `wind_loads` | Out of geotechnical scope (ASCE 7-22 wind) | — |

Deleted directories (confirmed gone): `pyseismosoil_agent/`, `fdm2d/`,
`geolysis_agent/`, `wind_loads/`.

## Files edited (de-wiring)

### Packaging / build
- **`pyproject.toml`**
  - `[project] description`: "36 analysis modules" → "32 analysis modules".
  - `[project.optional-dependencies]`: removed `pyseismosoil = ["PySeismoSoil>=0.5"]`
    and `geolysis = ["geolysis>=0.4"]`; removed `"PySeismoSoil>=0.5"` + `"geolysis>=0.4"`
    from the `full` list. (`fdm2d`/`wind_loads` had no extras.)
  - `[tool.setuptools.packages.find] include`: removed the 4 globs
    (`pyseismosoil_agent*`, `geolysis_agent*`, `wind_loads*`, `fdm2d*`).
  - `[tool.pytest.ini_options] testpaths`: removed the 4 entries.
- **`.github/workflows/tests.yml`**: removed the CI `pip install PySeismoSoil`
  and `pip install geolysis` lines (optional-dep best-effort installs).

### funhouse_agent
- **`funhouse_agent/adapters/__init__.py`**: de-registered the 4 entries from
  `MODULE_REGISTRY` (`pyseismosoil`, `geolysis`, `wind_loads`, `fdm2d`).
- **`funhouse_agent/adapters/`** — deleted: `pyseismosoil_adapter.py`,
  `fdm2d_adapter.py`, `geolysis.py`, `wind_loads.py`.
- **`funhouse_agent/dispatch.py`**: removed the 3 `_METHOD_ALIASES` entries for
  `fdm2d`, `wind_loads`, `pyseismosoil`. (`ANALYSIS_MODULES`/`REFERENCE_MODULES`
  are derived from `MODULE_REGISTRY`, so they updated automatically: now 32 + 21.)
- **`funhouse_agent/native_tools.py`**: NO CHANGE NEEDED — its 7 tool schemas are
  generic (no per-module schemas/dispatch).
- **`funhouse_agent/system_prompt.py`**: NO CHANGE NEEDED — module catalog +
  counts are generated from `MODULE_REGISTRY` at runtime (auto-updated).
- **`funhouse_agent/geotech_test_suite.json`**: removed the eval questions for the
  removed modules (WL-1, GL-1, GL-2, PSS-1, FDM-1). JSON re-validated.
- **`funhouse_agent/tests/test_phase34_adapters.py`**: dropped the FDM2D section
  (TestFdm2dMethodInfo/Dispatch/Calls) + docstring line.
- **`funhouse_agent/tests/test_new_adapters.py`**: dropped the pyseismosoil
  section (TestPyseismosoil*), the `"pyseismosoil"` parametrize entry, and updated
  "12 adapters" → "11 adapters".
- **`funhouse_agent/tests/test_method_name_aliases.py`**: dropped the fdm2d /
  wind_loads / pyseismosoil ROUTING_CASES and the
  `test_wind_loads_alias_reaches_real_method` test.

### foundry
- **`foundry/`** — deleted: `pyseismosoil_agent_foundry.py`,
  `geolysis_agent_foundry.py`, `wind_loads_agent_foundry.py`.
- **`foundry/__init__.py`**: NO CHANGE — it is a docstring-only package marker
  (no registry to de-register from).
- **`foundry/bearing_capacity_agent_foundry.py`**: removed the two `geolysis_agent.*`
  cross-refs from the `related` map and the two geolysis steps from `typical_workflow`.
- **`foundry/subsurface_char_agent_foundry.py`**: removed the `geolysis.classify_uscs`
  cross-ref from a `related` map.

### foundry_test_harness
- **`scenarios.py`**: removed the now-unused geolysis-only scenarios
  (`SPT_CORRECTION`, `USCS_CL`, `USCS_SW`, `AASHTO_A7`, `CROSS_CHECK_BEARING`).
- **`test_tier1_textbook.py`**: removed the geolysis + wind_loads imports, the
  `TestGeolysis` and `TestWindLoads` classes, and their two `TestMetadata`
  parametrize tuples.
- **`test_tier2_workflows.py`**: removed the geolysis import; reworked
  `TestFoundationDesignWorkflow` so the bearing→settlement workflows no longer
  chain geolysis classify/SPT steps; deleted the geolysis-only
  `test_spt_to_bearing_capacity`.
- **`test_tier3_crosscheck.py`**: removed the geolysis import; deleted the
  geolysis-vs-bearing `test_vesic_bearing_capacity`, the geolysis-only
  `test_spt_overburden_correction`, and the geolysis-only
  `TestClassificationCrossCheck`; reworked `test_spt_bearing_cross_check` →
  `test_spt_to_friction_angle` (keeps the groundhog half).
- **`test_tier4_error_handling.py`**: removed the geolysis import + its entry in
  `ALL_AGENTS`; deleted the geolysis-only `test_spt_missing_eop`.

### Docs / coordination ledgers
- **`README.md`**: intro "35 analysis modules" → "32"; "Core Analysis (20)" →
  "(19)" (dropped `wind_loads` row); "Library Wrapper Agents (15)" → "(13)"
  (dropped `pyseismosoil_agent` + `geolysis_agent` rows); dropped `pyseismosoil`
  + `geolysis` from the Optional Extras table.
- **`CLAUDE.md`**: intro module list; inventory table (dropped the 4 rows);
  counts updated — "57 modules = 36 analysis + 21 reference" → "53 = 32 + 21";
  funhouse "52 modules / 36 analysis adapters" → "48 / 32"; "adapters/ 52 (36+16)"
  → "48 (32+16)"; "improves the 37 analysis modules" → "32". (The dated
  "`fdm2d` wall-clock guard" line in a historical Done-note was left as a record.)
- **`docs/funhouse_agent_guide.md`**: module-catalog tables (dropped the 4 rows);
  counts "50 modules" → "46", "Core Analysis (18)" → "(17)", "External Library
  Adapters (7)" → "(6)", "FEM/FDM & Visualization (3)" → "FEM & Visualization (2)";
  adapters-tree comment counts.
- **`docs/funhouse_agent_expansion_plan.md`**: added a consolidation note at top;
  adjusted phase/totals counts (Phase 1 +6, Phase 3 +2, Total 46); added a
  "Consolidation −2" row; "50 modules" → "46". (Original phase text kept as
  historical record, annotated.)
- **`docs/agent_showcase.html`**: removed the Geolysis + PySeismoSoil table rows
  and the two standalone domain-card showcase blocks.
- **`module_work/BOARD.md`**, **`foundations.md`**, **`slope-fem.md`**,
  **`triage_feedback.py`**: removed the 4 modules from the domain rosters /
  owned-module lists / backlog lines / the `DOMAINS` dict, and trimmed two
  historical-triage module lists that named them.

### Exception edits (kept files that referenced a removed module)

Neither matched the exact wording in the checklist, but each was surgically cleaned:
- **`dashboard_app.py`**: the `MODULE_META` display list (NOT an import — there was
  no `wind_loads` import section) listed `pyseismosoil_agent` and `geolysis_agent`.
  Removed both rows. (It did not reference `fdm2d` or `wind_loads`.)
- **`geotech_qt_gui.py`**: a single About-dialog help string listed `wind_loads`
  among example modules (NOT an `fdm2d` import as the checklist anticipated).
  Removed `wind_loads` from that string. No code/import dependency existed.

## Left as-is (regenerable artifacts / historical records — NOT wiring)

These still contain the module names but are generated data snapshots or dated
changelog entries, not live wiring. Editing them would desync computed values or
rewrite history:
- **`dashboard.html`** — fully generated statistics snapshot (computed bar widths,
  percentages, line/test counts). Regenerate via `dashboard_app.py` if desired.
- **`docs/geotech_test_suite_results.json`**, **`module_work/module_feedback.json`**
  — captured Databricks eval-run outputs (point-in-time data).
- **`reference_work/BOARD.md`** — a dated "fdm2d hang fix" changelog line (record
  of past work).
- The dated "Done" notes in `CLAUDE.md` / expansion-plan phase text that mention
  the removed modules historically.

## Test results

Command: `.venv/Scripts/python.exe -m pytest -q` (worktree root on sys.path first).

- `funhouse_agent` + `foundry_test_harness`: **666 passed, 5 skipped, 0 failed**
  (100 s). The 5 skips are pre-existing optional-dep skips.
- Full configured suite (all kept modules; `geotech-references` submodule
  excluded per DO-NOT-TOUCH): see commit-time run — green, no failures introduced.
- Post-removal import check: `MODULE_REGISTRY` = 53 (32 analysis + 21 reference),
  none of the 4 removed names present, `_METHOD_ALIASES` clean, `system_prompt`
  + `native_tools` import fine.

## Later phases pending (NOT done here)

- **`DM7Eqs/` deletion needs consumer migration first.** Three kept consumers import
  `plot_figure_*` from `DM7Eqs` and must be re-pointed to `geotech_references.dm7_*`
  before `DM7Eqs/` can be removed:
  - `settlement/calc_steps.py:421`
  - `calc_package/tests/test_calc_package.py`
  - `dashboard_app.py:135`
- **opensees lateral-pile retirement** (harvest the above-ground stickup feature
  into native `lateral_pile` first; keep opensees for site response + PM4Sand).
- **Unified liquefaction API** over SPT + CPT. Side bug to fix: `seismic_geotech`
  docstrings/`__init__` mis-cite "Boulanger & Idriss (2014)" but the code actually
  implements the **NCEER/Youd-2001 SPT** procedure — correct the citation.
- **Data-parser merge**: `subsurface_characterization` + `pygef_agent` +
  `ags4_agent` + `pydiggs_agent` → one unified data-I/O module.
- **GUI retirement**: Dash (`slope_stability_gui.py`, `fem2d_gui.py`) + Qt
  (`geotech_qt_gui.py`, `slope_stability_qt.py`, `fem2d_qt.py`, `qt_panels/`) +
  the `gui` optional-dep.
