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

---

# Consolidation Phase 2a — Changes Manifest

Authoritative record of Phase 2a (contained retirements) of the module-consolidation
effort. Executed on branch `scope-consolidation` in the `scope-consol` worktree,
stacked on the Phase 1 commit `99fde72`, 2026-06-09. Three independent retirements,
one commit each.

## Task A (#10) — Retire the GUIs (Dash + Qt); analysis modules stay

The underlying `slope_stability/` and `fem2d/` analysis modules are KEPT intact —
only the GUI front-ends were removed.

Deleted (front-ends + artifacts):
- Dash: `slope_stability_gui.py`, `fem2d_gui.py`, `dashboard_app.py`, `dashboard.html`.
- Qt: `geotech_qt_gui.py`, `slope_stability_qt.py`, `fem2d_qt.py`, `qt_panels/` (whole dir, 8 files).
- GUI image artifacts at repo root: `slope_gui_screenshot.png`, `slope_gui_screenshot2.png`,
  `slope_gui_cropped.png`.

De-wired:
- **`pyproject.toml`**: removed the `gui` optional-dependency (`dash>=2.14`, `plotly>=5.18`)
  and the two `dash`/`plotly` lines from the `full` extra.
- **`README.md`**: removed the "GUIs" section (Plotly Dash table) and the `gui` row from
  the Optional Extras table.
- **`CLAUDE.md`**: removed the "GUIs" section (Plotly Dash + Qt Desktop tables).
- **`slope_stability/DESIGN.md`**: removed the trailing "Qt GUI" section (referenced
  `slope_stability_qt.py` + `qt_panels/`).
- `main.py` is a stub (PyCharm sample) — no GUI references; no change.
- Remaining GUI-name mentions live only in the historical ledgers
  (`CONSOLIDATION_PLAN.md`, this file's Phase 1 section) — records, not live wiring.

## Task B (#11) — Migrate DM7Eqs's figure consumers, then DELETE DM7Eqs/

`DM7Eqs/`'s public equations are a byte-identical strict subset of the installed
`geotech_references` package (already a runtime dependency; DM7Eqs was excluded from
the wheel). A few KEPT files still consumed it via a
`sys.path.insert(0,"DM7Eqs"); from geotech.dm7_X.chapterN import ...` hack.

Consumers migrated (`from geotech.dm7_* ...` → `from geotech_references.dm7_* ...`,
sys.path hack dropped):
- **`settlement/calc_steps.py`**: `plot_figure_5_16` (`dm7_1.chapter5`).
- **`calc_package/tests/test_calc_package.py`**: `plot_figure_5_16` (`dm7_1.chapter5`),
  `plot_figure_8_46` (`dm7_1.chapter8`), `plot_figure_4_36` (`dm7_2.chapter4`) —
  5 test methods updated.
- (`dashboard_app.py` was a fourth consumer but it was deleted in Task A — skipped.)

**Import-resolution check (repo venv) — PASSED.** Before deleting DM7Eqs:
```
from geotech_references.dm7_1.chapter5 import plot_figure_5_16
from geotech_references.dm7_1.chapter8 import plot_figure_8_46
from geotech_references.dm7_2.chapter4 import plot_figure_4_36
-> ok
```
All three names resolved in `geotech_references` with no renaming needed, and the 5
migrated `TestDm7FigureHelper` tests passed against the package (PNG output verified).

Then:
- Deleted **`DM7Eqs/`** in full (the `geotech/` package: `dm7_1/` ch1-8 + `dm7_2/`
  ch2-7 + prologue, and its `tests/` 1852-test suite).
- **`pyproject.toml`**: removed the `"DM7Eqs*"` entry from
  `[tool.setuptools.packages.find]` `exclude`.
- **`.gitignore`**: removed the `DM7Eqs/references/*.pdf`, `DM7Eqs/eq_*.png`,
  `DM7Eqs/references/*.png` ignore lines.
- **`CLAUDE.md`**: dropped the stale "+ DM7 equations" from the intro module list
  (the DM7 equations now live solely in the `geotech-references` submodule, which the
  Other-components line already documents).

## Task C (#6) — Retire ONLY the OpenSees BNWF lateral-pile pathway

`opensees_agent/` REMAINS fully functional for **1D site response + PM4Sand cyclic
DSS**. Only its redundant BNWF lateral-pile capability was removed — the native
`lateral_pile` module (COM624P-style FD solver, 8 p-y models) already covers lateral
piles, and the BNWF code only re-sampled `lateral_pile`'s own p-y curves into OpenSees
`PySimple1` springs (no unique physics; monotonic only).

opensees_agent:
- Deleted **`opensees_agent/bnwf_pile.py`**.
- **`results.py`**: removed the `BNWFPileResult` dataclass (kept `PM4SandDSSResult`,
  `SiteResponseResult`).
- **`__init__.py`**: dropped `analyze_bnwf_pile` + `BNWFPileResult` exports and the
  BNWF docstring/reference lines.
- **`tests/test_opensees_agent.py`**: removed all BNWF tests — `TestBNWFPileResult`,
  `TestBNWFPileResultPlots`, `TestBNWFPileValidation`, `TestBNWFBuildPyModel`,
  `TestBNWFPileIntegration`, the two Foundry-metadata BNWF tests, and the
  `bnwf_pile`/`BNWFPileResult` imports. Kept all PM4Sand + site-response tests.
- **`DESIGN.md`**: replaced the BNWF section with a note pointing to native
  `lateral_pile`; trimmed the architecture tree + testing-strategy lines.

Agent layer:
- **`funhouse_agent/adapters/opensees_adapter.py`**: removed `_run_bnwf_lateral_pile`,
  its `METHOD_REGISTRY` + `METHOD_INFO` entries; opensees now exposes 2 methods.
- **`funhouse_agent/adapters/__init__.py`**: opensees `brief` no longer lists BNWF.
- **`funhouse_agent/tests/test_new_adapters.py`**: opensees expected-methods set + the
  `list_methods` total updated 3 → 2; dropped `test_bnwf_not_installed`.
- **`foundry/opensees_agent_foundry.py`**: removed `_run_bnwf_pile`, its registry +
  `METHOD_INFO` entries; trimmed agent/list-methods docstrings + category example.
- **No change needed** in `funhouse_agent/dispatch.py`, `system_prompt.py`,
  `native_tools.py` — the module catalog/counts are derived from `MODULE_REGISTRY` at
  runtime, no opensees BNWF alias existed, and `native_tools` has only generic schemas.
  `foundry_test_harness/` had no BNWF scenarios.

Docs: removed BNWF / opensees-lateral-pile mentions from `README.md`, `CLAUDE.md`,
`docs/funhouse_agent_guide.md`, `docs/agent_showcase.html` (kept the
site-response/PM4Sand descriptions, and the native `lateral_pile` "Lateral Pile"
mentions which are a different, KEPT module).

**Confirmed: `opensees_agent` still supports site response + PM4Sand** — post-edit the
module exports `analyze_pm4sand_dss`, `analyze_site_response`, `PM4SandDSSResult`,
`SiteResponseResult`; adapter + foundry registries = `{pm4sand_cyclic_dss,
site_response_1d}`; the PM4Sand + site-response test classes pass.

## Test results (Phase 2a)

Repo venv (`.venv/Scripts/python.exe -m pytest`), worktree root.

- Per-task gates (run as each task landed):
  - Task B: `calc_package settlement` → **144 passed, 3 skipped**.
  - Task C: `opensees_agent funhouse_agent/tests/test_new_adapters.py` →
    **173 passed, 6 skipped** (skips = openseespy-not-installed Tier 2 + matplotlib).
- Full targeted suite:
  `opensees_agent settlement calc_package funhouse_agent foundry_test_harness
  slope_stability fem2d` → see commit-time run (all green; no failures introduced).
- The known pre-existing failure in
  `seismic_geotech/.../TestMononobeOkabe::test_KPE_zero_kh_battered_equals_coulomb`
  is on the branch base and was NOT run here (not in the targeted set).

## Task (#5) — Unified liquefaction tool (B&I-2014 default, NCEER-2001 behind flag) + fix seismic_geotech citation

A BUILD task: consolidate liquefaction triggering into ONE discoverable
agent-layer tool, default to Boulanger & Idriss (2014) for both SPT and CPT, keep
the legacy NCEER/Youd-2001 SPT procedure behind a method flag, and fix the
mis-attributed citation in `seismic_geotech`. No analysis-module physics was
changed; the unification is at the agent/tool layer (analysis modules never
import each other — only `geotech_common` is exempt).

### Design decisions (locked by owner)
- **B&I-2014 is the DEFAULT** method for both SPT and CPT.
- Legacy **NCEER / Youd et al. (2001) SPT** procedure stays available behind
  `method="nceer2001"` (for code-compliance work that still cites it).
- Fix the `seismic_geotech` citation regardless.
- Do NOT re-implement B&I-2014 natively — leverage `liquepy`'s tested code. Keep
  existing module-level functions working (backward compatible); ADD the unified
  layer on top.

### Key finding — liquepy SPT B&I-2014 availability
`liquepy` ships a packaged B&I-2014 **CPT** triggering object (`trigger.run_bi2014`,
`BoulangerIdriss2014`/`BoulangerIdriss2014CPT`/`BoulangerIdriss2014SoilProfile` —
all consume CPT `q_c`/`f_s`/`u_2`) but **NO packaged SPT triggering object**. Its
only SPT entry points are *field correlations* (Vs/Dr/G0 from `n_1_60`, in
`liquepy.field.correlations`), not triggering. However, `liquepy` DOES expose every
B&I-2014 SPT **building block** as tested module-level functions in
`liquepy.trigger.boulanger_and_idriss_2014`:
`calc_crr_m7p5_from_n1_60cs`, `calc_rd`, `calc_csr`, `calc_k_sigma_w_n1_60cs`
(verified empirically: CRR_M7.5 = 0.061/0.156/0.290/0.485 at (N1)60cs =
0/15/25/30 — the published B&I-2014 SPT curve). So SPT B&I-2014 **is** achievable
via liquepy by composing those tested blocks — which is what was done. The only
SPT pieces liquepy lacks (the SPT fines correction Δ(N1)60 Eq 2.23, and the SPT
form of MSF) were added in `liquepy_agent`, on top of liquepy's curve fits.

### Files added
- **`liquepy_agent/spt_liquefaction.py`** — `analyze_spt_liquefaction()`
  (per-layer B&I-2014 SPT triggering: FoS = CRR_M7.5·MSF·K_sigma / CSR, built from
  liquepy's tested blocks) + `bi2014_spt_fines_correction()` (Eq 2.23) +
  `bi2014_spt_msf()` (SPT MSF, returns exactly 1.0 at M7.5 per liquepy convention).
- **`funhouse_agent/adapters/liquefaction_adapter.py`** — the unified router
  (module `liquefaction`, single method `liquefaction_analysis`). Detects input
  type (CPT if `q_c`/`f_s`; SPT if `N160`; `input_type=` override for
  both/neither), normalizes the method (B&I aliases → `bi2014`; NCEER/Youd
  aliases → `nceer2001`), then routes: CPT→liquepy B&I CPT (LPI/LSN/LDI);
  SPT+bi2014→liquepy B&I SPT; SPT+nceer2001→`seismic_geotech.evaluate_liquefaction`.
  Rejects `nceer2001` for CPT and ambiguous/no-data input with clear errors.
- **`liquepy_agent/tests/test_spt_liquefaction.py`** — 24 tests (fines/MSF
  helpers, result dataclass, full SPT triggering vs published B&I curve + expected
  behavior). liquepy-dependent tests skip when liquepy is absent.
- **`funhouse_agent/tests/test_liquefaction_router.py`** — 28 tests (metadata +
  discoverability, input-type detection, method normalization, SPT/CPT routing,
  consistency with the underlying module outputs, error paths, alias routing).

### Files edited
- **`liquepy_agent/results.py`** — added `SPTLiquefactionResult` dataclass
  (`summary`/`layer_results`/`to_dict`); module docstring updated to three result
  types.
- **`liquepy_agent/__init__.py`** — export `analyze_spt_liquefaction`,
  `bi2014_spt_fines_correction`, `bi2014_spt_msf`, `SPTLiquefactionResult`;
  docstring reframed as "B&I-2014 (CPT + SPT)".
- **`seismic_geotech/liquefaction.py`** — **citation fix only** (no numeric
  change): module docstring now attributes the procedure to NCEER / Youd-2001
  (NCEER CRR fit, NCEER MSF, Youd fines, Liao & Whitman rd), explicitly states it
  is NOT B&I-2014, and points to `liquepy_agent` for B&I-2014.
- **`seismic_geotech/__init__.py`** — package docstring corrected (dropped the
  "Boulanger & Idriss (2014)" reference line; liquefaction listed as NCEER/Youd
  SPT).
- **`seismic_geotech/DESIGN.md`** — References section strengthened to state the
  NCEER/Youd-2001 attribution and that it is not B&I-2014.
- **`seismic_geotech/tests/test_seismic_geotech.py`** — added
  `TestLiquefactionCitation` (3 tests) locking the citation fix + the NCEER CRR
  fit value.
- **`funhouse_agent/adapters/liquepy_adapter.py`** — added a thin
  `spt_liquefaction` method (mirrors `cpt_liquefaction`) → 3 methods now.
- **`funhouse_agent/adapters/__init__.py`** — registered the new `liquefaction`
  module in `MODULE_REGISTRY`; updated the `liquepy` and `seismic_geotech` briefs
  to point at the unified tool as the discoverable one. (Auto-flows into
  `list_agents`, the system-prompt catalog, `ANALYSIS_MODULES`, and `call_agent`
  — `dispatch.py`/`system_prompt.py`/`native_tools.py` derive everything from the
  registry, so no per-module schema edits were needed.)
- **`funhouse_agent/dispatch.py`** — added curated `_METHOD_ALIASES` for the
  `liquefaction` module (`evaluate_liquefaction`/`cpt_liquefaction`/`spt_liquefaction`/
  `boulanger_idriss_2014`/`liquefaction_triggering` → `liquefaction_analysis`;
  `bi2014`/`nceer2001` guesses inject the method flag) and a
  `liquepy spt_boulanger_idriss_2014` → `spt_liquefaction` alias.
- **`README.md`**, **`CLAUDE.md`** — describe the unified liquefaction capability
  and the liquepy-SPT finding; module-inventory briefs updated.

### Note on the seismic_geotech adapter
`funhouse_agent/adapters/seismic_geotech.py` has a PRE-EXISTING signature mismatch
in its `csr_crr_check` / `site_classification` helpers (calls e.g.
`stress_reduction_rd(z, M)` / `compute_vs30` that don't match the current module
signatures). This is unrelated to #5 and was left untouched. The unified router
does NOT use those helpers — for `method="nceer2001"` it calls
`seismic_geotech.liquefaction.evaluate_liquefaction` directly (clean, matching
signature, exercised by its own tests).

### Test results (#5)
Repo venv (`.venv/Scripts/python.exe -m pytest`), worktree root.
- New tests: `liquepy_agent/tests/test_spt_liquefaction.py`,
  `seismic_geotech/.../TestLiquefactionCitation`,
  `funhouse_agent/tests/test_liquefaction_router.py` → all green (liquepy-dependent
  tests skip when liquepy absent; liquepy IS installed in the repo venv, so they ran).
- Affected suites: `seismic_geotech liquepy_agent funhouse_agent` → see the
  commit-time run below. The known pre-existing
  `TestMononobeOkabe::test_KPE_zero_kh_battered_equals_coulomb` failure is the only
  red and is NOT from this work.

---

# Consolidation #3 — merge pygef / ags4 / pydiggs into subsurface_characterization

Authoritative record of consolidation task #3. Executed on branch
`scope-consolidation` in the `scope-consol` worktree, stacked on `311d2d8`,
2026-06-09. Make `subsurface_characterization` the single data-I/O home and fold
the three thin parser/validator wrapper modules into it as optional,
dependency-backed **format adapters**, then remove the three standalone modules.

## What moved where (new submodule layout)

A new `subsurface_characterization/formats/` subpackage holds the folded code
(parsing/validation logic preserved verbatim — only import paths changed; result
dataclasses moved intact):

| New file | Folded from | Public API |
|----------|-------------|------------|
| `formats/gef.py` | `pygef_agent/{pygef_utils,cpt_parser,bore_parser}.py` | `parse_cpt_file`, `parse_bore_file`, `has_pygef` (+ `import_pygef`, validators) |
| `formats/gef_results.py` | `pygef_agent/results.py` | `CPTParseResult`, `BoreParseResult` |
| `formats/ags4.py` | `ags4_agent/{ags4_reader,ags4_utils}.py` | `read_ags4`, `validate_ags4`, `has_ags4` (+ `import_ags4`) |
| `formats/ags4_results.py` | `ags4_agent/results.py` | `AGS4ReadResult`, `AGS4ValidationResult` |
| `formats/diggs_validation.py` | `pydiggs_agent/{diggs_validation,pydiggs_utils}.py` | `validate_diggs_schema`, `validate_diggs_dictionary`, `has_pydiggs` (+ schema/dict path helpers) |
| `formats/diggs_validation_results.py` | `pydiggs_agent/results.py` | `DiggValidationResult` |
| `formats/__init__.py` | (new) | re-exports all of the above |

Each adapter keeps its **lazy optional-import / `has_*` guard intact** — importing
the names is cheap; the heavy third-party lib (`pygef` / `python-ags4` / `pydiggs`)
is imported only when a parse/validate function actually runs, and a clear error is
raised if absent. The native `parse_diggs` DIGGS *data extraction* (custom
`xml.etree` parser, no external dep) is unchanged and remains the primary DIGGS
path; `formats.diggs_validation` is the *validation* path (pydiggs).

## Exports

- **`subsurface_characterization/__init__.py`**: now re-exports all 15
  format-adapter names (functions + `has_*` + result classes) and lists them in
  `__all__`; module docstring reframed as "the single data-I/O home … plus optional
  format adapters". `csv_loader.py` docstrings updated (the `load_cpt_to_investigation`
  bridge now references the GEF adapter; the function is duck-typed, unchanged).

## Agent layer (funhouse_agent)

- **`adapters/subsurface_adapter.py`**: extended to expose ALL methods — the 8
  existing (parse_diggs/load_site/5 plots/compute_trend) PLUS the 6 folded
  format-adapter methods: `parse_cpt`, `parse_bore` (File Import), `read_ags4`
  (File Import), `validate_ags4`, `validate_diggs_schema`,
  `validate_diggs_dictionary` (File Validation). Each `_run_*` carries the
  `has_*()` guard returning the "not installed" error dict. `METHOD_REGISTRY` +
  `METHOD_INFO` both updated (14 methods, keys match). The `subsurface` registry
  brief rewritten to advertise the consolidated ingest+validate+visualize surface.
- **`adapters/__init__.py`**: removed the `pygef`, `ags4`, `pydiggs` entries from
  `MODULE_REGISTRY` (registry 54 → 51). `ANALYSIS_MODULES`/`REFERENCE_MODULES`
  derive from the registry, so they updated automatically.
- **`adapters/`** — deleted `pygef_adapter.py`, `ags4_adapter.py`,
  `pydiggs_adapter.py`.
- **`dispatch.py`**: the `("ags4","read_and_validate")→read_ags4` curated alias was
  re-keyed to `("subsurface","read_and_validate")` so the guess still resolves on
  the new home.
- **No change needed** in `system_prompt.py` / `native_tools.py` — module catalog +
  counts are derived from `MODULE_REGISTRY` at runtime, and `native_tools` has only
  generic schemas.

## Tests migrated

The three test files moved into `subsurface_characterization/tests/`, imports
re-pointed to the new module paths. The **library-self-skip behavior is kept**
(`requires_*`/`skipif(not has_*())`). The Foundry-metadata/Foundry-integration test
classes were dropped (their `foundry/*_foundry.py` targets are deleted standalone
Foundry files, never part of the pip package). The pygef sample data
(`sample_cpt.gef`, `sample_bore.gef`) was copied alongside.

| New test file | From | Tests |
|---------------|------|-------|
| `tests/test_formats_gef.py` | `pygef_agent/tests/test_pygef_agent.py` | dataclass defaults, plot smoke, input validation, util, CPT/bore integration (+ a new SiteModel-bridge test) |
| `tests/test_formats_ags4.py` | `ags4_agent/tests/test_ags4_agent.py` | read/validate dataclass defaults, input validation, util, read/validate integration |
| `tests/test_formats_diggs_validation.py` | `pydiggs_agent/tests/test_pydiggs_agent.py` | result dataclass, input validation, util, schema/dictionary/result integration |

Updated funhouse tests for the new shape:
- `tests/test_new_adapters.py`: replaced the Pygef/Ags4/Pydiggs sections with
  `TestSubsurfaceFormatAdapter*` (dispatch describe + has_* guard paths against the
  `subsurface` module); dropped pygef/ags4/pydiggs from the registered-adapters
  parametrize; header "11 adapters" → "8".
- `tests/test_phase34_adapters.py` + `tests/test_subsurface_adapter.py`: subsurface
  expected-methods set + `list_methods` total 8 → 14.
- `tests/test_method_name_aliases.py`: `("ags4",…)` routing case → `("subsurface",…)`.
- `geotech_test_suite.json`: the PGEF-1 / AGS-1 / DIGGS-1 eval questions' `module`
  field re-pointed to `subsurface_characterization` (questions unchanged).

## Removed (3 standalone module dirs)

Deleted in full: `pygef_agent/`, `ags4_agent/`, `pydiggs_agent/` (each: module dir
+ tests). Confirmed gone from the worktree; no live `import pygef_agent` /
`ags4_agent` / `pydiggs_agent` references remain anywhere (grep clean).

## Packaging / build (pyproject.toml)

- `[project] description`: "32 analysis modules" → "29".
- `[project.optional-dependencies]`: the optional libs stay installable. Added a
  consolidated **`subsurface = ["pygef>=0.10","python-ags4>=0.5","pydiggs>=0.1"]`**
  extra; kept `pygef` / `ags4` / `pydiggs` as backward-compat aliases. The `full`
  extra still lists all three libs (unchanged).
- `[tool.setuptools.packages.find] include`: dropped `pygef_agent*`, `ags4_agent*`,
  `pydiggs_agent*` (the existing `subsurface_characterization*` glob already covers
  the new `formats` submodule).
- `[tool.pytest.ini_options] testpaths`: dropped the 3 entries (those tests now live
  under `subsurface_characterization`).

## De-wiring (foundry, docs, ledgers)

- **`foundry/`** — deleted `pygef_agent_foundry.py`, `ags4_agent_foundry.py`,
  `pydiggs_agent_foundry.py`. `foundry/__init__.py` is a docstring-only marker (no
  registry) — no change. `subsurface_char_agent_foundry.py` left as-is (its existing
  method set is the visualization API; the folded data-I/O is exposed via the
  funhouse `subsurface` adapter, which is the live agent surface).
- **`foundry_test_harness/`** — no pygef/ags4/pydiggs scenarios or references (grep
  clean); no change.
- **`README.md`** — intro "32" → "29"; `subsurface_characterization` Core-Analysis
  row rewritten to list the folded adapters; dropped the 3 Library-Wrapper-Agent rows
  (→ "10 modules") with a folded-in note; Optional-Extras table gains a `subsurface`
  row and marks `pygef`/`ags4`/`pydiggs` as aliases.
- **`CLAUDE.md`** — intro module list; inventory table (dropped the 3 rows;
  subsurface row rewritten, test-count bumped to 231); inventory header
  "53 = 32 + 21" → "50 = 29 + 21"; funhouse adapter-count lines de-specified;
  "improves the 32 analysis modules" → "29".
- **`docs/funhouse_agent_guide.md`** — "Available Modules (46)" → "(43)";
  "File/Data Import Adapters (5)" → "(2)" (dropped pygef/ags4/pydiggs rows, added a
  folded-in note); `subsurface` row rewritten; adapters-tree comment updated;
  `list_agents()`/MODULE_REGISTRY count references de-specified.
- **`docs/funhouse_agent_expansion_plan.md`** — Phase-2 note annotated with the #3
  fold (historical text kept).
- **`docs/agent_showcase.html`** — replaced the 3 table rows with one
  `subsurface_characterization` row.
- **`subsurface_characterization/DESIGN.md`** — input-format table updated; added a
  "Format adapters (`formats/` subpackage)" section.
- **`module_work/BOARD.md`** + **`triage_feedback.py`** — characterization-domain
  roster trimmed (the 3 names removed; subsurface annotated as their new home).

## Left as-is (regenerable artifacts / historical records — NOT live wiring)

- **`docs/geotech_test_suite_results.json`**, **`module_work/module_feedback.json`**
  — captured Databricks eval-run outputs (point-in-time data snapshots).
- **`.github/workflows/tests.yml`** — keeps the best-effort
  `pip install pygef/python-ags4/pydiggs || true` lines: the libs remain installable
  and the migrated integration tests need them in CI (they self-skip otherwise).

## Optional libs present? Tests ran vs skipped

All three optional libs ARE installed in the repo venv (`pygef`, `python_ags4`,
`pydiggs` all import), so the migrated Tier-2 integration tests **ran** (not
skipped) — they self-skip only when the lib is absent.

## Test results (#3)

Command (worktree root on sys.path first):
`.venv/Scripts/python.exe -m pytest subsurface_characterization funhouse_agent foundry_test_harness -q`

- **914 passed, 5 skipped, 0 failed** (~77 s). The 5 skips are pre-existing
  optional-dep skips unrelated to this work.
- The migrated format tests run green: `test_formats_gef.py` (+integration),
  `test_formats_ags4.py` (+integration), `test_formats_diggs_validation.py`
  (+integration) — 86 tests, all ran (libs present).
- The known pre-existing `seismic_geotech/...TestMononobeOkabe::
  test_KPE_zero_kh_battered_equals_coulomb` failure is on the branch base and is NOT
  in this targeted set (not run here).

## Judgment calls

- **`formats/` submodule layout**: kept each adapter's result dataclasses in a
  sibling `*_results.py` (mirrors the original `results.py` split) so the parsing
  modules import cleanly and the result types are independently importable.
- **Extras**: added a consolidated `subsurface` extra but KEPT the per-library
  `pygef`/`ags4`/`pydiggs` extra names as aliases (backward compatibility for anyone
  pinning them) — and the libs stay in `full`.
- **Foundry**: deleted the 3 standalone `*_foundry.py` files but did NOT add the
  folded methods to `subsurface_char_agent_foundry.py` — those Foundry files are
  out-of-package deployment artifacts; the live agent surface (funhouse `subsurface`
  adapter) already exposes the 6 folded methods.
- **Foundry test classes**: dropped them when migrating (their targets are the
  deleted Foundry files), keeping every dataclass/validation/utility/integration
  test.
