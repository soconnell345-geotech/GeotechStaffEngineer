# GeotechStaffEngineer

Python toolkit for LLM-based geotechnical engineering agents.
31 analysis modules (incl. pavement_design, AASHTO 1993) + groundhog wrapper + OpenSees agent + pyStrata agent + seismic signals agent + liquepy agent + hvsrpy agent + GSTools agent + SALib agent + swprocess agent + pystra agent + subsurface characterization (DIGGS/GEF/AGS4 data I/O — folds in the former pygef/ags4/pydiggs wrappers as format adapters) + DXF import + DXF export + PDF import + fem2d (2D plane-strain FEM: T6 quadratic elements, 3D-principal MC return, GL99 strength reduction, staged construction) + reliability (FOSM/PEM/Monte Carlo/native FORM + published COV database) + geo_project (staged, human-gated LLM model setup) + funhouse_agent (engine-agnostic agent with vision).

## What this is for (framing — use this voice in user-facing docs)

Geotechnical engineering is the *practice* of building **on and in the ground** —
foundations, retaining walls, slopes, excavations, embankments. The engineer's
material is the earth itself: heterogeneous, layered, partly saturated, and
sampled at only a handful of points across a site. Because the ground is variable
and only partly known, **a single number is never the answer**. Geotechnical
analysis is **repeated calculation across plausible subsurface and loading
conditions** — running a chained assortment of industry-standard formulas,
empirical and numerical methods, then comparing against the **design
requirements** and varying assumptions until the design is robust across the
uncertainty. The engineer's job is to understand the **range and spread** of
answers, not one point estimate; the **true answer is a distribution**.

This toolkit packages those methods as clean, machine-callable Python
(deterministic `analyze_*()` functions → dataclasses), wraps them in a
probabilistic variability engine (`reliability/` — FOSM/PEM/MC/FORM → β, P_f),
and drives them with an engine-agnostic LLM agent (`funhouse_agent/`). It follows
the lineage of every tool that amplified the engineer — slide rule → spreadsheet
/ FEM → Monte Carlo → LLM agent — **not a replacement for judgment, but a
multiplier for it.** (Plain-language framing lives in `README.md` /
`docs/overview.html`; keep that practitioner voice — "practice," "on and in the
ground," "design requirements," "subsurface and loading conditions" — in
user-facing copy.)


## Architecture Patterns

Every analysis module follows this structure:
```
module_name/
  __init__.py          # exports analyze_*() + result classes
  <domain>.py          # core computation functions
  results.py           # @dataclass with summary() -> str, to_dict() -> dict
  tests/
    test_<module>.py   # pytest suite
  DESIGN.md            # theory, sign conventions, edge cases (read when working on this module)
```

Key conventions:
- **All units SI**: meters, kPa, kN, kN/m, degrees
- **Dict-based I/O** for LLM agents: analyze_*() returns dataclass, .to_dict() for JSON
- **No cross-module imports** between analysis modules (geotech_common is the exception)
- **SoilProfile adapters** in `geotech_common/soil_profile.py` bridge SoilProfile -> module inputs
- **Foundry wrappers** (`foundry/` dir + `geotech-references/agents/`): 34 + 14 = 48 agents, 3 functions each (agent/list/describe). These are standalone Foundry deployment files, NOT part of the pip package.

## v5.8.1 status (RELEASED 2026-07-17 to PyPI; owner OK'd — Foundry-deployment fixes)

Patch driven by the owner's first live Foundry publish (same night as 5.8.0):
(1) custom-model-RID clobber FIXED — the keyed Model selectbox's sticky state
reverted programmatic model changes (custom RID box AND conversation resume)
within one render; `_model_dirty` flag now syncs the widget pre-instantiation
(AppTest regression tests in webapp/tests/test_custom_model_rid.py). (2)
Extras consolidation (owner request): plain `pip install geotech-staff-engineer`
now brings the WHOLE stack (deep agent, webapp, all backends, PDF, both LLM
clients incl. langchain-openai); the 23 old extra names remain as empty
aliases. (3) `websockets>=14,<16` pin — Foundry's Streamlit base image ships
websockets 16.x, which leaked into the app lockfile and made the publish-time
reinstall unsatisfiable vs langgraph-sdk (<16). Also: eval suite now 108 Q /
68 keyed (PAV-1..PAV-8, ground truth run on v5.8.0), eval_harness `--ids`
prefix filter, description says 31 modules. Foundry ops notes: published-app
"failed to run startup scripts" = read the env-restore log (the websockets
conflict presented there); preview needs `streamlit run <file> --server.port
8501` in the terminal first. Single-namespace package restructure (kill the 35
top-level modules) assessed and PARKED as the 6.0.0 candidate.

## v5.8.0 status (RELEASED 2026-07-17 to PyPI; owner OK'd — UFC alternative method + pavement specialist)

UFC 3-250-01 (2016) roads/parking design as a full alternative method in
`pavement_design/ufc.py`: `design_flexible_pavement_ufc` (Corps CBR / Figure E-1
cover cascade + Table 7-2 minimums + Ch 19 reduced-subgrade-strength frost),
`design_rigid_pavement_ufc` (Figure F-1 Westergaard-based + Eq 13-1 stabilized
foundation), `compare_flexible_pavement_methods` (AASHTO vs UFC side by side,
FHWA Mr=2555·CBR^0.64 bridge, caveats echoed), `ufc_mixed_traffic` (Table G-1
controlling-vehicle equivalent-18-kip passes; light vehicles below controlling
thickness → unlimited passes; reproduces printed G-1 within 6%). Plots module
(nomograph-overlay design charts, layer sections, UFC charts, method
comparison); calc-package design_type flexible_ufc/rigid_ufc/compare; pavement
specialist agent (app picker + reviewer + Claude Code twin). F-1 stands at
exact-at-anchors ±10% — no printed equation exists; N-densification provably
fails (documented in refs docstring). Ships with **geotech-references 1.3.3**
(pin >=1.3.3): ufc_pavement REBUILT from the real UFC 3-250-01 (closing the old
airfield-source audit gap) + all 30 Appendix E vehicle curves + 3 companion
practice modules (refs now 27 modules). Airfield (UFC 3-260-02) and rigid
vehicle curves F-2..F-31 PARKED.

## v5.7.0 status (RELEASED 2026-07-16 to PyPI; owner OK'd — pavement design)

New `pavement_design/` analysis module (31st): complete AASHTO 1993 flexible
(Fig 3.1 SN + Fig 3.2 layer split + §3.1.4 minimums + forward check) and rigid
(Fig 3.7; k = direct | MR/19.4 | full §3.2 composite-k worksheet, LS-corrected,
iterated with D) design, ESAL traffic (full Appendix D LEF tables incl. triple
axles), Appendix G swelling/frost-heave serviceability loss (printed G.4/G.8
equations) + Table 3.1 performance-period iteration, calc-package template +
`pavement_design_package`, adapter with 5 methods. US customary (documented
exception). Validation V-055 (guide worked examples SN 5.0 / D 10.0 / MR 5000
end-to-end); DESIGN.md carries the full ledger + chart-read tolerances. Ships
with **geotech-references 1.3.2** (pin >=1.3.2): EC7-1/EC7-2/AASHTO-1993 +
lef.py (~5,850 cells, lead visual QC) + composite_k.py (read-grid digitization)
+ environmental.py. The 1993 Guide is a SINGLE volume — the D-2 "Appendix MM of
Volume 2" citation is stale 1986 text (owner-verified). Overlays (Part III)
PARKED. list_agents catalog at 7,995/8,000 chars — rebalance briefs before
adding the next module.

## v5.6.0 status (RELEASED 2026-07-15 to PyPI; owner OK'd — the app train)

Owner pivot to app-heavy work (`webapp/`); wiki integration PARKED. All A-items
A1–A8 built: inline Plotly, calc sub-agent context isolation
(default ON, −84% measured), crash-proof turn persistence, durable files/links,
Agent + Analysis-depth + model pickers, per-conversation working folder
(`GEOTECH_DEFAULT_OUTPUT_DIR`), optional tracing (`GEOTECH_TRACE=1` / LangSmith
envs), industry memo. Owner-session fixes: bounded auto-continue for
mid-turn stops, download MIME types, recursion-cap visibility. Also in 5.6.0:
generic `calc_package.html_to_pdf` (verified HTML→PDF for reports no canned
package covers) + calc-agent REAL-DISK-ONLY framing; retaining_walls
`delta_base`/`base_adhesion` direct base-interface overrides + `sliding_basis`
echo (double-2/3 trap resolved — verdict in retaining_walls/DESIGN.md) +
widened `retaining_wall_package` params; **Palantir Foundry deployment path**
(`docs/FOUNDRY.md`: Code-Workspaces publish, LLM-proxy engine with `ri.…` RID
auto-routing OpenAI/Anthropic, `GEOTECH_FOUNDRY_MODELS`, in-app custom-RID box,
`webapp/foundry_entry.py` 2-line stub). Owner decisions: summarization backstop
SKIPPED, durable checkpointer PARKED, API thinking layer DEFERRED, `foundry/`
AIP-wrapper route RETIRED. Plans: `module_work/APP_PLAN.md`,
`module_work/FOUNDRY_APP_PLAN.md`. Webapp tests: 115 (`pytest webapp/tests -q`).

## v5.5.0 status (RELEASED 2026-07-12 to PyPI; owner OK'd)

The post-5.4.1 train, merged and released; close-out gate **8539 passed /
48 skipped**. Adds: correlated scalar-pair probabilistic variables
(`correlations=[(k1,k2,rho)]` on fosm_fos/monte_carlo_fos, V-045; overlapping
pairs rejected), Bray & Travasarou (2007) seismic displacement
(`bray_travasarou_2007`, paper anchors reproduced + hand-verified),
Lowe-Karafiath rapid drawdown (`method='lowe_karafiath'`, V-048/052/053;
"Corps #1/#2" = interslice-function menu-name collision, intentionally NOT a
drawdown method), anisotropic undrained strength (`strength_model='anisotropic'`,
su_active/su_dss/su_passive, V-054, exact isotropic identity), the one-call
`slope_report_package` (search story + rejection diagnostics + method table +
thrust line + FOSM/MC annex; hardened: param rejection, renderer surfaced,
Story PDFs ~1 MB, wide tables fit, thrust-line spike fixed source+display),
and native inline Plotly in `webapp/` (.plotly.json sidecar). Lead close-out
QC ledger: `module_work/V5.4_PLAN.md` §"Close-out QC" (7 findings, all fixed).
Deferred: toe-circle search under-sampling; steep-φ' Kc sensitivity.
5.4.1 (2026-07-10) carried the file round-trip (read_pdf_text, list_files,
verified saves, plot output_path, Attach widget, PDF fallback) + `webapp/`.

## v5.4.0 status (RELEASED 2026-07-08 to PyPI; owner OK'd pre-live-eval)

Merged to master and released; release gate **8436 passed / 48 skipped**.
On top of the six owner directives (below), 5.4.0 carries the FULL E1–E11 QC
backlog (rapid-drawdown search + validated #98, stage-3 'gle' option — default
STAYS 'fellenius' per owner, pore-pressure TIN grids, exit-side/truncation
tension cracks — V-026 <0.1%, composite pile EI — V-017 PASS, pile passive
convention, SRM mesh-refinement study, multi-surcharge zones, perf cleanups,
eval suite 71→**100** keyed questions, cross-module did-you-mean redirects,
fem2d `analyze_footing_capacity` — Prandtl −0.4%, ufc_expansive figures
complete in geotech-references ≥1.3.1) and F1/F2/F5/F8 (correlated `linear_su`
+ `ru`/`gamma_sat` probabilistic variables — Duncan Pf anchor, V-042/043/044,
search-admissibility guard — zero-shift proven, local-FOS heatmap + gallery
Exhibit 13, reviewer family: seismic/foundations/earth-retention/slope-fem).
DEFERRED to next train: correlated c'-φ' scalar-pair + Slide2 #33/#34, F3
Bray-Travasarou, F4 CoE/Lowe-Karafiath, F6 anisotropic su, F7 slope
calc-package. Owner-gated: live 100-Q eval run.

All six owner directives DONE (2026-07-07):
**D1** 132-pp PDF user manual (`docs/GeotechStaffEngineer_User_Manual_v5.3.pdf`,
regenerable builder, catalog auto-generated from METHOD_INFO); **D2** Databricks
no-restart flow (`funhouse_agent/runtime_check.py` hot-reloads a stale
`typing_extensions` before langchain imports); **D3** layered disclaimers
(DISCLAIMER.md ships in the wheel, PyPI README section, one-time first-import
notice w/ `GEOTECH_NO_DISCLAIMER=1`, `geotech-disclaimer` script, calc-package
basis blocks — pip runs no code on wheel install, these are the equivalents);
**D4** 12-exhibit visualization gallery (`docs/gallery/`, real validated runs);
**D5** `drawing_ir/` module (LLM-ready drawing IR: exact-coordinate entities w/
provenance + confidence, DXF / PDF-vector / raster-opencv legs, agent query
surface via handles; `[raster]` extra; geo_project wiring = flagged follow-up);
**D6** seismic reviewer (`.claude/agents/seismic-reviewer.md` +
`funhouse_agent.make_seismic_reviewer(engine)` / `make_seismic_reviewer_deep`;
shared checklist `funhouse_agent/review_checklists.py` — reviewer-family
template). Open backlog: `module_work/V5.4_PLAN.md` E1–E11 (QC carryovers) +
F1–F8 (creative proposals). **HANDOFF.md §3 is the authoritative v5.4 summary.**

## v5.3.0 status (RELEASED 2026-07-06 to PyPI; geotech-references stays 1.3.0)

**5.3.0 adds** (on top of the 5.2.0 line): Batch-2 coverage 5/5 (drilled_shaft rational
GEC-10 chains, MSE LRFD external-stability CDRs, soe basal-heave-sidewall-shear + FHWA
apparent-pressure anchored walls + log-spiral Caquot-Kerisel Kp, full Reese-1974 sand
p-y, fem2d monolithic Taylor-Hood u-p Biot consolidation); slope_stability round 2
(15 new Slide2/ACADS/Duncan validation problems V-026..V-040, noncircular-search
robustness fix + rejection diagnostics, rapid drawdown 2/3-stage, Newmark + Jibson,
infinite slope, Ito-Matsui stabilizing piles verified vs the original 1975 paper);
pdf_import round 2 (scale calibration, label→region, cleanup, vision grid overlay,
vision↔vector cross-check); and 12 adversarial-review fixes (headline: log-spiral Kp
δ=0 Rankine anchor). Plan of record + deferred follow-ups: `module_work/V5.3_PLAN.md`.
The owner-gated publish rule still applies to FUTURE releases: no version bump / tag /
publish without explicit owner OK (a `v*` tag push auto-publishes via
.github/workflows/publish.yml).

**>> For a full cross-session handoff read `HANDOFF.md` (repo root) first. <<**

Everything since v5.0.0 is summarized in `docs/V5.1_SUMMARY.html` (one page; validation
tables). Master carries: the v5.1 backlog (token round-cap, calc-QC round 3, adapter
ergonomics, eval answer-keys/sample files, foundry audit, references review); the LE+FEM
modernization (branches `le-modern`/`fem-modern`, merged); the `reliability/` module
(consolidation #7 DONE); calc-package figures/tables + plotly interactive viewers
(`calc-viz`); the staged model-setup agent (`geo_project/` + `deep/setup_agent.py`,
OFF by default via `build_deep_agent(enable_setup_agent=True)`); the post-v5.0
lateral-pile calc-package fix + adapter-ergonomics sweep + the Databricks /Workspace
placeholder-write fix (`funhouse_agent/_fileio.py`); **Phase E** published-example
validation (25 problems as 87+ offline tests in `validation_examples/`, `RESULTS.md`,
0 analysis bugs -- also added general fem2d `roller_base` BC + `initial_stress_relaxation`);
and **v5.2 coverage Batch 1** (four additive, default-preserving capability adds:
`settlement` Hough, `pile_group` Meyerhof group settlement, `axial_pile` per-layer
`toe_friction_angle` + `head_depth`, `retaining_walls` MSE bar-mat Kr/F* curves -- see
`module_work/V5.2_COVERAGE.md`; Batch 2 = the bigger builds, NOT yet started).
Intentional behavior changes vs 5.0 (battered-wall KPE, pile_group +Mx sense, sheet-pile
embedment basis, wave-equation damping default, fem2d T6 default) are listed in the
summary page.
Databricks install: from /tmp or a UC Volume, `%pip install "geotech-staff-engineer[deep,full]"`
(the `[full]` extra covers the optional analysis backends — without it ~12 of the 97 eval
questions fail honestly with "not installed"), then
`dbutils.library.restartPython()` -- a stale runtime `typing_extensions` (<4.13) otherwise
breaks langgraph imports with "unexpected keyword argument 'extra_items'". Funhouse health
check: `funhouse_agent/deep/rc_wheel_check.py` (`run_rc_check(fh_prompter)`); the 100-question eval
suite: `funhouse_agent/deep/eval_harness.py` (`run_suite(model, out=...)`). Save outputs to
`/tmp` or `/Volumes`, NOT `/Workspace` (FUSE writes are non-durable / permission-blocked).

## Module Inventory (31 analysis + geo_project setup layer + 24 reference, + foundry harness; reference layer fully QC'd, all figure catalogs 100% page-accurate)

| Module | Tests | Purpose |
|--------|-------|---------|
| bearing_capacity | 83 | Shallow foundations (CBEAR/Vesic/Meyerhof, load-spread two-layer, GWT-in-wedge) |
| settlement | 53 | Consolidation & immediate (CSETT, Schmertmann, shape-factored elastic) |
| axial_pile | 65 | Driven pile capacity (Nordlund/Tomlinson/Beta, GWT-split integration, uplift) |
| sheet_pile | 36 | Cantilever/anchored walls (Rankine/Coulomb w/ wall friction; single FOS basis) |
| soe | 111 | Support of excavation (braced/cantilever, stability, anchors) |
| lateral_pile | 12+97 | Lateral pile (COM624P p-y models, banded FD solver, above-ground stickup; validation.py oracle suite) |
| pile_group | 85 | Rigid cap groups (6-DOF, one RH sign convention end-to-end, Converse-Labarre) |
| wave_equation | 60 | Smith 1-D wave equation (elasto-plastic springs, smith/smith_viscous damping, bearing graph) |
| drilled_shaft | 60 | GEC-10 alpha/beta/rock socket (clay end-bearing cap, N60 side reduction) |
| seismic_geotech | 82 | Site class, M-O pressures (battered-wall-correct), Fpga(PGA), SPT liquefaction (NCEER/Youd-2001) |
| retaining_walls | 93 | Cantilever + MSE walls (GEC-11; thrust decomposition, coherent-gravity MSE, Meyerhof bearing) |
| pavement_design | 42+ | AASHTO 1993 pavement design (US customary): flexible SN + Fig 3.2 layer split, rigid slab D (direct/MR-19.4/composite-k), ESAL traffic (Appendix D LEFs), calc-package template; orchestrates geotech_references.aashto_1993 |
| ground_improvement | 49 | Aggregate piers (incl. Priebe n0), wick drains, surcharge, vibro (GEC-13) |
| slope_stability | 384+17skip | Rigorous GLE/M-P (Fredlund-Krahn) + Bishop/Janbu/Spencer/OMS, entry-exit + DE noncircular search, nails/anchors/geosynthetics, SHANSEP/Hoek-Brown, ponded water, probabilistic FOS (FOSM/MC), SLOPE/W-grade plots; validated vs F&K-1977/ACADS/Duncan (VALIDATION.md) |
| reliability | 176 | Probabilistic geotech engines (FOSM/PEM/Monte Carlo/native FORM), published COV knowledge base (Duncan 2000/TC304/Phoon-Kulhawy), Vanmarcke spatial averaging, bearing/pile/slope wrappers |
| downdrag | 53 | Fellenius neutral plane, UFC 3-220-20 downdrag |
| geotech_common | 288 | SoilProfile (82) + checks (93) + adapters (89) + plots (21) |
| opensees_agent | 106 | PM4Sand cyclic DSS, 1D site response |
| pystrata_agent | 60 | 1D EQL site response (SHAKE-type, Darendeli/Menq/custom) |
| seismic_signals | 74 | Earthquake signal processing (eqsig/pyrotd) |
| liquepy_agent | 59 | Boulanger & Idriss (2014) liquefaction triggering — CPT (LPI/LSN/LDI) + SPT |
| hvsrpy_agent | 37 | HVSR site characterization from ambient noise |
| gstools_agent | 69 | Geostatistical kriging, variogram fitting, random fields |
| salib_agent | 35 | Sobol & Morris sensitivity analysis |
| swprocess_agent | 30 | MASW surface wave dispersion analysis |
| pystra_agent | 43 | FORM/SORM/Monte Carlo structural reliability analysis |
| subsurface_characterization | 231 | Subsurface data I/O: DIGGS parser (20 test types) + Plotly plots + trend stats; PLUS folded format adapters — GEF/BRO-XML CPT/borehole parse (pygef), AGS4 read/validate (python-ags4), DIGGS schema/dictionary validation (pydiggs) |
| dxf_import | 97 | DXF CAD import for slope stability + FEM (discover layers, parse geometry, build SlopeGeometry/FEM inputs) |
| dxf_export | 37 | DXF export for cross-section geometry (surface, boundaries, GWT, nails, annotations) |
| pdf_import | 56 | PDF cross-section import (PyMuPDF vector extraction, LLM vision extraction, geometry conversion) |
| drawing_ir | 61 | LLM-ready drawing digitization ("Drawing IR"): deterministic extractor owns coordinates, LLM owns semantics. Unified JSON IR (Line/Polyline/Arc/Circle/Text/Region w/ coords+layer/color+provenance+per-entity confidence) from DXF (ezdxf, conf 1.0), PDF vector (pdf_import+scale module, conf 1.0), raster (NEW OpenCV Hough/contour tracing, conf <1); slice-query interface (bbox/angle/text/layer/nearest) so the LLM requests slices, not pixels |
| fem2d | 353 | 2D plane-strain FEM (T6 default + CST/Q4/beam, 3D-principal MC return, HS, GL99 SRM, seepage, consolidation, staged construction, PLAXIS-style calc-package plots); validated vs Griffiths-Lane/Prandtl (VALIDATION.md) |
| geo_project | 89 | Canonical Project document for staged, human-gated LE/FEM model setup (schema+validators, builders, templates, DXF/PDF/vision ingest w/ provenance quarantine, echo-back renderer) |

Other components: groundhog_agent (90 methods), geotech-references submodule (382 DM7 + 95 GEC/micropile + 10 FEMA + 9 NOAA + 35 UFC functions + DM7 figure catalogs, 3529 tests), foundry_test_harness (142 tests), funhouse_agent (106 + 149 + 163 + 25 + 31 + 5 = 479 tests)

## Foundry Test Harness

`foundry_test_harness/` validates all Foundry agent functions via JSON-in/JSON-out:

| File | Tests | Purpose |
|------|-------|---------|
| test_tier1_textbook.py | 73 | Individual functions vs textbook/published answers |
| test_tier2_workflows.py | 14 | Multi-function engineering workflows (e.g., classify → SPT → bearing → settlement) |
| test_tier3_crosscheck.py | 10 | Cross-agent consistency (same problem, different agents) |
| test_tier4_error_handling.py | 41 | Bad JSON, unknown methods, missing params, invalid values |

Supporting files: `harness.py` (FoundryAgentHarness class), `scenarios.py` (reusable problem definitions)

Run: `pytest foundry_test_harness/ -v`

## PDF Import (Cross-Section Geometry Extraction)

`pdf_import/` extracts cross-section geometry from PDF drawings using two methods:

1. **Vector extraction** — PyMuPDF `page.get_drawings()` for exact geometry
2. **Vision extraction** — LLM image analysis via pluggable `image_fn`

| File | Purpose |
|------|---------|
| `__init__.py` | Exports + `to_dxf_parse_result()` adapter |
| `results.py` | `PdfParseResult` dataclass (mirrors DxfParseResult fields) |
| `extractor.py` | PyMuPDF vector path extraction + `discover_pdf_content()` |
| `vision.py` | LLM vision extraction + JSON parsing |
| `tests/` | 56 tests (programmatic PDFs + mock vision functions) |

Workflow: `discover_pdf_content()` → `extract_vector_geometry()` → `to_dxf_parse_result()` → `build_slope_geometry()` / `build_fem_inputs()`

Requires: `PyMuPDF >= 1.23` (optional: `pip install geotech-staff-engineer[pdf]`)

Run: `pytest pdf_import/ -v`

## Funhouse Agent (Engine-Agnostic Geotechnical Agent)

`funhouse_agent/` provides an engine-agnostic geotechnical agent with text + vision capabilities. Works with any AI backend satisfying the `GenAIEngine` protocol. Self-contained dispatch layer routes tool calls directly to the analysis + reference modules via internal adapters — no dependency on `foundry/` files. Analysis-module adapters + 16 geotech-references adapters (DM7 340+ equations, 7 GEC/micropile references with text retrieval, FEMA P-2192, NOAA frost, 4 UFC standards, the cross-reference text-search DB `reference_db`, and the figure-catalog search DB `figure_db` — which pairs with the `read_reference_figure` vision tool to find an engineering chart by meaning and read a value off it). The `subsurface` adapter now also exposes the folded GEF/AGS4/DIGGS-validation format-adapter methods (parse_cpt/parse_bore/read_ags4/validate_ags4/validate_diggs_schema/validate_diggs_dictionary).

| File | Purpose |
|------|---------|
| `__init__.py` | Exports: `GeotechAgent`, `GenAIEngine`, `ClaudeEngine`, `NativeToolEngine`, `PrompterBridgeEngine`, `USING_SDK_ENGINES`, `AgentResult` |
| `engine.py` | `GenAIEngine` Protocol + engines. **Hybrid**: prefers the Funhouse SDK's `funhouse.services.prompter.engine` (`NativeToolEngine`/`PrompterBridgeEngine`) when importable, falls back to local classes (`USING_SDK_ENGINES` flag). Local `ClaudeEngine` retained. |
| `agent.py` | `GeotechAgent` (native + text-ReAct loops, vision dispatch, `allowed_agents` scoping, `reference_mode` + `consult_references`) |
| `dispatch.py` | Tool dispatch + shared `REFERENCE_MODULES`/`ANALYSIS_MODULES` constants + `allowed_agents` scoping |
| `reviewer.py` | Reference consult sub-agent (`consult_references`) + legacy post-hoc reviewer (`run_review`/`needs_revision`) |
| `system_prompt.py` | Self-contained system prompt (50 modules) |
| `native_tools.py` | OpenAI tool schemas + dispatch for NativeToolEngine |
| `vision_tools.py` | Vision tools: `analyze_image`, `analyze_pdf_page`, `read_reference_figure` (render a catalogued figure + read a value off it), `save_file` |
| `notebook.py` | `NotebookChat` — ipywidgets chat interface for Jupyter/Databricks |
| `adapters/` | analysis-module adapters + 16 reference adapters bridging flat JSON → module APIs (the former pygef/ags4/pydiggs adapters were folded into `subsurface_adapter`) |
| `tests/` | 106 tests (mock engines, no API key needed) |

Usage:
```python
from funhouse_agent import GeotechAgent, ClaudeEngine, NativeToolEngine

# With PrompterAPI (Databricks) — native OpenAI tool calling (recommended)
agent = GeotechAgent(genai_engine=NativeToolEngine(fh_prompter))

# With PrompterAPI — text-based ReAct (legacy, may not work with newer GPT models)
agent = GeotechAgent(genai_engine=fh_prompter)

# With Claude
agent = GeotechAgent(genai_engine=ClaudeEngine())

result = agent.ask("Calculate bearing capacity of 2m footing, phi=30")

# Interactive notebook chat (ipywidgets)
from funhouse_agent.notebook import NotebookChat
chat = NotebookChat(agent)
chat.display()
```

Run: `pytest funhouse_agent/ -v`

### Unified liquefaction tool (`liquefaction`)

Liquefaction triggering is consolidated into ONE discoverable agent-layer tool,
`liquefaction` (adapter `adapters/liquefaction_adapter.py`, method
`liquefaction_analysis`). It auto-routes by **input type** and **method** (no
cross-module imports — the routing lives at the agent layer, which is allowed to
call multiple modules):

- **CPT** (`q_c`/`f_s` present) → `liquepy` Boulanger & Idriss (2014) CPT
  (`liquepy_agent.analyze_cpt_liquefaction`), with LPI / LSN / LDI.
- **SPT** (`N160` present), `method="bi2014"` (**DEFAULT**) → `liquepy` B&I-2014
  SPT (`liquepy_agent.analyze_spt_liquefaction`).
- **SPT**, `method="nceer2001"` → legacy NCEER / Youd et al. (2001) simplified
  procedure (`seismic_geotech.evaluate_liquefaction`), for code-compliance work.

B&I-2014 is the default for both SPT and CPT. The per-module functions and the
direct `liquepy`/`seismic_geotech` adapters remain intact and callable.

**liquepy SPT note:** `liquepy` ships a packaged B&I-2014 CPT triggering object
(`run_bi2014`) but **no packaged SPT triggering object** — its only SPT entry
points are *field correlations* (Vs/Dr/G0 from N). It DOES expose every B&I-2014
SPT *building block* as tested module-level functions
(`calc_crr_m7p5_from_n1_60cs`, `calc_rd`, `calc_csr`, `calc_k_sigma_w_n1_60cs`),
so `liquepy_agent.analyze_spt_liquefaction` composes those into a full SPT
triggering procedure (adding only the B&I SPT fines correction Eq 2.23 and the
SPT MSF, which liquepy lacks for SPT). **seismic_geotech citation fix:** its
SPT procedure is NCEER / Youd-2001 (NCEER CRR fit, NCEER MSF, Youd fines, Liao &
Whitman rd) — the docstrings that previously cited "Boulanger & Idriss (2014)"
were corrected; the numerical code was unchanged.

### Reference consult-agent (`reference_mode`)

The primary agent no longer calls the 21 reference modules directly — reference access is routed
through a single `consult_references` tool backed by a **reference-scoped sub-agent** (a
`GeotechAgent` restricted to `REFERENCE_MODULES` via `allowed_agents`, generalized from
`reviewer.py`; uses the same engine, so it gets native tool calling). `GeotechAgent(reference_mode=...)`:
- `"anytime"` (**default**) — `consult_references` always offered; the primary is auto-scoped to
  the analysis modules (`ANALYSIS_MODULES`), shrinking its tool surface (which cut method/module-
  name guessing in live testing).
- `"after_calc"` — `consult_references` offered only after a `call_agent` has run this `ask()`.
- `"off"` — legacy: the reference modules stay directly callable, no consult tool.

`allowed_agents` (new `GeotechAgent` param) scopes the system-prompt catalog AND
`list_agents`/`list_methods`/`describe_method`/`call_agent`. `REFERENCE_MODULES` (21) and
`ANALYSIS_MODULES` (36) are shared constants in `dispatch.py`. The consultant holds `figure_db` +
`read_reference_figure`, so it does figure read-off too. Verified live in Databricks (5.1 deployment).

## Module-Improvement Agent Team

A standing, domain-organized agent team improves the 30 analysis modules over time, fed by the
agent test-suite feedback and other tasks. Claude Code teammates are ephemeral, so the team's
identity and memory live as **version-controlled files**:

- `.claude/agents/geotech-team-lead.md` — the team-lead playbook (run by the main session):
  triage feedback, maintain the board, dispatch specialists, review every diff, gate on tests,
  serialize `geotech_common` edits.
- `.claude/agents/geotech-module-specialist.md` — the reusable specialist (spawn one per domain,
  named by domain, e.g. `seismic`): reads its ledger → edits its module + adapter → runs
  `pytest <module>/ -v` → updates its ledger → returns a diff (does NOT commit; the lead reviews).
- `module_work/BOARD.md` — backlog + status across all domains; the single source of truth, with
  the triaged feedback categories and the 28-clean / 40-recovered / 0-failed agent-suite baseline.
- `module_work/<domain>.md` — per-domain progress ledgers (owned modules, reference map, backlog,
  log). A specialist reads its ledger at task start and updates it at the end.
- `module_work/triage_feedback.py` + `module_feedback.json` — turn a Databricks
  `geotech_test_suite_results.json` run into per-domain/per-module work orders.
- `funhouse_agent/geotech_test_suite.json` + `funhouse_agent/smoke_test_native.py` — the
  68-question eval set and the Databricks smoke/probe script that generate the feedback.

Domains (in the board): foundations, deep-foundations, earth-retention, slope-fem, seismic,
characterization, io-cad, references, common (lead-serialized).

**Status: the module-fix backlog is CLEARED (v5.1, 2026-06-10).** Phase 0 ergonomics, the
allowed_values rollout (23 adapters), the param-name hotspot fixes, and calc-QC Round 3 are all
on master; `module_work/BOARD.md` and `module_work/calc_qc/FINDINGS.md` carry the close-out logs.
The board remains the template for future feedback-driven rounds. `.claude/agents/` is
committed (the repo `.gitignore` was changed to `.claude/*` + `!.claude/agents/`).

## Reference-Layer Build Work (figure catalogs + new references)

A separate ongoing effort builds/QCs the geotech-references library. Infrastructure:
- `.claude/agents/figure-catalog-builder.md` — reusable subagent that drives any catalog's
  `page_estimated → 0` (finds each figure's true PDF page via caption/footer/List-of-Figures
  search) and builds catalogs for new references; can fan out one worker per reference.
- `reference_work/BOARD.md` — running ledger of the reference build/QC backlog + status.

**Working model (user preference, 2026-06-05): autonomous, milestone-level — NOT per-step.**
Drive the multi-agent work to completion; pick sensible defaults for ordering / which-first and
proceed without asking; commit + push freely as milestones land; report at milestones. The user
reviews this module in big batches (~weekly), not step-by-step. **No version bumps / PyPI
publishes until the user explicitly says so.** Still surface genuinely consequential / irreversible
decisions. See memory `feedback-reference-layer-autonomy`.

**Done (2026-06-05, released as geotech-references 1.2.5 / geotech-staff-engineer 4.6.5):**
- Deleted low-value/incorrect refs: `noaa_frost`, `ufc_dewatering`, `fema_p2192`.
- Built `ufc_backfill` + `ufc_expansive` figure catalogs; **ALL 20 figure catalogs at 100% page
  accuracy** (`page_estimated=0`).
- Three NEW references, full pipeline (text JSON + python lookups + figure catalog + tests +
  registry wiring in both repos): **`fema_p2082`** (FEMA P-2082 / 2020 NEHRP — site classes
  BC/CD/DE with BC baseline, Fa/Fv removed), **`california_trenching`** (Caltrans T&S Manual —
  shoring/excavation), **`fhwa_pavements`** (FHWA-NHI-05-037 — Mr/CBR/frost/drainage; distinct
  from `ufc_pavement`). Reference layer is now **24 modules** (EC7-1/EC7-2/AASHTO-1993 added 2026-07-15) (`dispatch.REFERENCE_MODULES`).
- **Deep QC** (4-agent fan-out) found & fixed 3 critical + 4 major content/functional errors
  (incl. a broken figure `pdf_path` that disabled vision for 2 refs, and the FEMA Ch-19 SSI
  equations that a build agent had mis-reconstructed); ~640 reference methods verified 0-bug.
- **Ergonomics:** semantic aliases for reference methods (`_reference_common.register_semantic_aliases`)
  + consult round-budget fix; **smart method resolution + analysis-method aliases** in `dispatch.py`
  (`_METHOD_ALIASES`, selector-value directives, fuzzy did-you-mean) driven by the agent-suite triage
  (`module_work/module_feedback.json`). (A `fdm2d` wall-clock guard was added here too; `fdm2d`
  has since been removed in the Phase 1 consolidation — see `CONSOLIDATION_CHANGES.md`.)

**Open backlog:** the ~23 per-module param-name/value bugs and the `retaining_walls`
earth-pressure-coefficient gap were FIXED in v5.1 (adapter-ergonomics stream). Still open:
module-*selection* mis-routing (e.g. Rankine guessed on the wrong module) and a `fem2d`
footing-SRM convenience method.
`ufc_expansive` has ~22 more figures behind an OCR pass (scanned PDF). The weekend QC routine's
gec_6/7/12/13 chapter-text edits are a separate uncommitted workstream.

## HANDOFF — Figure Read-Off: Status & Remaining Work

> Handoff note for the next team (agentic pickup). The figure retrieval + vision
> read-off feature is **built, working, and committed**; what remains is listed
> as a TODO with priorities. Read this whole section before touching the figure
> subsystem, `reviewer.py`, `agent.py`, or `system_prompt.py`.

### Done (committed `fef450c` on `master`, pushed)
- **Autonomy gap CLOSED** — the agent now finds a chart via `figure_db.figure_search`
  **and renders + reads it** with `read_reference_figure`, instead of answering chart
  values from the caption + model memory. Verified live: DM7.2 **Figure 4-12**, passive
  coefficient **Kp ≈ 5.3** (correct ≈ 5.5; it previously confabulated ≈ 9.0 from memory).
- **The fix = "starve-the-shortcut signpost" (the agentic option, see below).** Each
  `figure_search` hit now carries a `read_value` next-step instruction pointing at
  `read_reference_figure` and forbidding from-caption/from-memory values. The search
  result identifies the figure but exposes no values to guess from, so the vision read-off
  is the path of least resistance. (`funhouse_agent/adapters/figure_db_adapter.py`)
- **Param robustness** — `figure_search` tolerates common result-cap aliases
  (`top_k`/`k`/`n`/`max_results`/…) instead of hard-crashing on an unexpected kwarg; and
  `extract_method_info` (`adapters/_reference_common.py`) no longer advertises
  `**kwargs`/`*args` catch-alls as required parameters.
- Routing fix (`read_reference_figure` in the agent dispatch tuple), strengthened tool #7
  wording, a local `ClaudeEngine` runner `funhouse_agent/run_local.py`, offline adapter
  tests (`tests/test_figure_db_adapter.py`), and an opt-in live end-to-end test
  (`tests/test_live_figure_readoff.py`, formerly `xfail`).

### Design context (why it's built this way)
- **Architecture: find-with-text, read-with-pixels.** Retrieve figures lexically (SQLite
  FTS5 over captions + chapter-cross-linked descriptions), then hand the *actual rendered
  page* to a vision model to read values. CLIP-style **image embeddings were rejected** —
  poor fit for axis-labeled line charts. Do not pursue image embeddings.
- **Autonomy approaches weighed:** *Soft* (prompt-only — rejected, unreliable); **Medium
  (agentic retrieval + starve-the-shortcut — CHOSEN)**; *Hard* (deterministic auto-render —
  rejected: discards the agent's judgment and over-fires the costly vision call on any
  question that merely mentions a figure). Medium keeps the agent *reasoning* while making
  the from-memory shortcut useless. **Caveat:** Medium makes cheating useless, not
  impossible — if the agent is ever still seen reporting chart values without rendering,
  escalate to the Hard backstop (TODO P4).

### TODO — remaining work (prioritized)
- [ ] **P1 — Hallucination-on-tool-error (concerning, general, NOT figure-specific).**
      When a tool call errors, the ReAct loop may **fabricate success** instead of
      retrying/reporting. Observed: after a failed `figure_search`, the agent invented
      "Figure 3-5, score 19.0" and wrote *"the search and figure read both completed
      successfully."* Fix in the ReAct loop (`funhouse_agent/agent.py`) and/or system
      prompt: a tool error MUST be retried with corrected args or reported — never
      reported as success. Consider a loop-level guard that blocks a final answer
      contradicting an error result.
- [~] **P2 — Partially addressed by the reference consult-agent.** The shared `REFERENCE_MODULES`
      set (now in `dispatch.py`) DOES include `figure_db`, and the consult sub-agent runs a full
      `GeotechAgent`, so it can `read_reference_figure`. Only the legacy post-hoc `reviewer.py`
      (`review=True`) still lacks `figure_db`/`read_reference_figure` — wire it there too if used.
- [x] **P3 — DONE (2026-06-03): catalog recipe rolled out to ALL 15 references** (2491 figures):
      DM7 (dm7_1/2) + GEC 4–14 + micropile + ufc_pavement. `build_figure_catalog.py` was
      generalized to handle dot/dash/spaced-dash ids, heading + heading-less (dotted-leader
      density) + no-List-of-Figures (body-caption extraction) + sequential "Figure N" numbering
      (manifest `"figure_numbering":"sequential"`) + multi-volume (manifest `"volumes"` list →
      per-figure `pdf_path`). `body_start` is derived from the LoF span, so a manifest needs only a
      `pdf_path` (or `volumes`). Released in geotech-references 1.2.3 / geotech-staff-engineer 4.6.3.
      Quality: most ≥94%; gec_8 37% and gec_11 42% are partial (correct figures, offset-estimated
      pages, flagged `page_estimated`).
- [ ] **P4 — (conditional) Hard/deterministic backstop.** Only if Medium proves
      insufficient in practice: a deterministic auto-render for detected chart-value
      questions. Do not build pre-emptively.
- [ ] **P5 — (optional) Lazy description enrichment.** On first read of a figure, capture
      a one-line semantic description + axis/variable list and write it back into
      `figures_catalog.json` to improve future `figure_search` recall at zero bulk cost.
      Planned in the original design, not built.
- [~] **P6 — Cheaper recall lever LANDED (2026-06-09); text embeddings still deferred.**
      Built the lexical **synonym query-expansion** layer first
      (`geotech_references/_query_expansion.py`) wired into `reference_search`/`figure_search`,
      selected by `EXPANSION_STRATEGY` (env `GEOTECH_RETRIEVAL_EXPANSION`, default `auto`).
      Eval (`scripts/eval_retrieval_recall.py`): recall@5 **11%→44%** with **0** top-1
      disturbance (`auto` = rerank the literal+synonym union but pin the literal top-1).
      27 new tests + full geotech-references suite (3702) pass. **Merged and live on master**;
      reaches the Funhouse consultant for free (shared `reference_db`/`figure_db` adapters).
      **v5.1 follow-ups:** live Funhouse reviewer-agent eval (owner-gated), synonym-map
      curation, larger gold set, keep-`auto`-default-vs-opt-in decision — see
      geotech-references `## Lexical Query Expansion`. **Text embeddings remain DEFERRED**
      (only if expansion+lexical still prove insufficient); **image embeddings remain rejected.**
- [ ] **P7 — (deferred) Figure cropping.** Full-page render is robust; revisit only if
      multi-figure pages hurt read-off accuracy.
- [x] **P8 — DONE: figure catalogs are packaged.** `"*/figures_catalog.json"` added to
      `geotech-references/pyproject.toml` `[tool.setuptools.package-data]`. So `figure_search`/
      `figure_get` work from a clean install. Released 1.2.1+.
- [x] **P9 — DONE: configurable PDF location.** `_figures_db.resolve_pdf()` now honors the
      `GEOTECH_REFERENCES_DOCS` env var (folder holding the source PDFs), falling back to the
      repo-relative `docs/`. On Databricks: copy a `docs/` folder of PDFs in and set that env var.
      PDFs are still NOT shipped in the wheel (large + license). Released 1.2.1+.

### Gotchas the next team MUST know
- **(RESOLVED 2026-06-03) The native-tool-calling work in `agent.py` & `system_prompt.py` is now
  COMMITTED** (`b163750`): the `## Available Modules` regex fix, the Tool Discipline section, and
  the reference consult-agent (`reference_mode`, `allowed_agents` scoping, `consult_references`).
  Confirmed working live in Databricks on the 5.1 deployment; released in 4.6.2/4.6.3.
- **Active adapter-generation churn.** `funhouse_agent/adapters/` is modified by a
  background process during sessions (new `gec*_adapter.py`, `ufc_pavement_adapter.py`,
  `adapters/__init__.py`, `tests/test_reference_adapters.py`, `module_work/`). Method
  counts can flip mid-run. **Stage by exact path and verify `git diff --cached` before
  committing.**
- **`geotech_references` is an editable install** (`pip install -e geotech-references/`),
  so reference/figure changes are live from source.
- **Live verification (opt-in, costs API).** Set `RUN_LIVE_TESTS=1` and provide
  `ANTHROPIC_API_KEY` (shell env or Windows *User* env), then:
  `.venv/Scripts/python -m pytest funhouse_agent/tests/test_live_figure_readoff.py -v -s`,
  or `.venv/Scripts/python -m funhouse_agent.run_local --demo`. The key is read at runtime
  from the Windows User env — never pass it through chat/transcripts.

## Working on a Module

1. Read the module's `DESIGN.md` first for theory and conventions
2. Read `__init__.py` for the public API
3. Run that module's tests: `pytest module_name/ -v`
4. Full regression: `pytest -q` (testpaths configured in pyproject.toml)

## Environment

- Windows 11, Python 3.14.3, venv at `.venv/`
- Git repo: github.com/soconnell345-geotech/GeotechStaffEngineer (private)
- Git submodule: `geotech-references/` → github.com/soconnell345-geotech/geotech-references (DM7 + future GEC refs)
- numpy >=2.0: use `np.trapezoid` (was `np.trapz`)
