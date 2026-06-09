# GeotechStaffEngineer

Python toolkit for LLM-based geotechnical engineering agents.
32 analysis modules + groundhog wrapper + OpenSees agent + pyStrata agent + seismic signals agent + liquepy agent + pygef agent + hvsrpy agent + GSTools agent + AGS4 agent + SALib agent + swprocess agent + pystra agent + pydiggs agent + subsurface characterization + DXF import + DXF export + PDF import + fem2d (2D plane-strain FEM with staged construction) + funhouse_agent (engine-agnostic agent with vision).

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

## Module Inventory (53 modules = 32 analysis + 21 reference, + foundry harness; reference layer fully QC'd, all figure catalogs 100% page-accurate)

| Module | Tests | Purpose |
|--------|-------|---------|
| bearing_capacity | 45 | Shallow foundations (CBEAR/Vesic/Meyerhof) |
| settlement | 39 | Consolidation & immediate (CSETT) |
| axial_pile | 55 | Driven pile capacity (Nordlund/Tomlinson/Beta) |
| sheet_pile | 26 | Cantilever/anchored walls (Rankine/Coulomb) |
| soe | 111 | Support of excavation (braced/cantilever, stability, anchors) |
| lateral_pile | 66 | Lateral pile (COM624P, 8 p-y models, FD solver) |
| pile_group | 72 | Rigid cap groups (6-DOF, Converse-Labarre) |
| wave_equation | 45 | Smith 1-D wave equation (bearing graph, drivability) |
| drilled_shaft | 48 | GEC-10 alpha/beta/rock socket |
| seismic_geotech | 71 | Site class, M-O pressures, liquefaction |
| retaining_walls | 70 | Cantilever + MSE walls (GEC-11) |
| ground_improvement | 43 | Aggregate piers, wick drains, surcharge, vibro (GEC-13) |
| slope_stability | 169+17skip | Fellenius/Bishop/Spencer, circular+noncircular, grid/random search, contact stresses, Duncan verification |
| downdrag | 53 | Fellenius neutral plane, UFC 3-220-20 downdrag |
| geotech_common | 288 | SoilProfile (82) + checks (93) + adapters (89) + plots (21) |
| opensees_agent | 106 | PM4Sand cyclic DSS, 1D site response |
| pystrata_agent | 60 | 1D EQL site response (SHAKE-type, Darendeli/Menq/custom) |
| seismic_signals | 74 | Earthquake signal processing (eqsig/pyrotd) |
| liquepy_agent | 59 | CPT-based liquefaction triggering (B&I 2014, LPI, LSN, LDI) |
| pygef_agent | 45 | CPT/borehole file parser (GEF/BRO-XML) |
| hvsrpy_agent | 37 | HVSR site characterization from ambient noise |
| gstools_agent | 69 | Geostatistical kriging, variogram fitting, random fields |
| ags4_agent | 39 | AGS4 geotechnical data format reader/validator |
| salib_agent | 35 | Sobol & Morris sensitivity analysis |
| swprocess_agent | 30 | MASW surface wave dispersion analysis |
| pystra_agent | 43 | FORM/SORM/Monte Carlo structural reliability analysis |
| pydiggs_agent | 31 | DIGGS 2.6 XML schema and dictionary validation |
| subsurface_characterization | 145 | Subsurface data visualization (DIGGS parser w/ 20 test types, Plotly plots, trend stats) |
| dxf_import | 97 | DXF CAD import for slope stability + FEM (discover layers, parse geometry, build SlopeGeometry/FEM inputs) |
| dxf_export | 37 | DXF export for cross-section geometry (surface, boundaries, GWT, nails, annotations) |
| pdf_import | 56 | PDF cross-section import (PyMuPDF vector extraction, LLM vision extraction, geometry conversion) |
| fem2d | 271 | 2D plane-strain FEM (CST/Q4/beam, MC/HS, SRM, excavation, pore pressures, seepage, consolidation, staged construction) |

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

`funhouse_agent/` provides an engine-agnostic geotechnical agent with text + vision capabilities. Works with any AI backend satisfying the `GenAIEngine` protocol. Self-contained dispatch layer routes tool calls directly to 48 modules (~890+ methods) via internal adapters — no dependency on `foundry/` files. Includes 32 analysis module adapters + 16 geotech-references adapters (DM7 340+ equations, 7 GEC/micropile references with text retrieval, FEMA P-2192, NOAA frost, 4 UFC standards, the cross-reference text-search DB `reference_db`, and the figure-catalog search DB `figure_db` — which pairs with the `read_reference_figure` vision tool to find an engineering chart by meaning and read a value off it).

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
| `adapters/` | 48 adapter modules (32 analysis + 16 reference) bridging flat JSON → module APIs |
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

A standing, domain-organized agent team improves the 32 analysis modules over time, fed by the
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

**Status: module-fix work is PINNED.** Phase 0 ergonomics (METHOD_INFO `allowed_values`, the
system-prompt Tool Discipline nudge, the native-catalog fix) is done; per-module fixes are paused
behind the reference-agent + figure work. Resume from `module_work/BOARD.md`. `.claude/agents/` is
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
  from `ufc_pavement`). Reference layer is now **21 modules** (`dispatch.REFERENCE_MODULES`).
- **Deep QC** (4-agent fan-out) found & fixed 3 critical + 4 major content/functional errors
  (incl. a broken figure `pdf_path` that disabled vision for 2 refs, and the FEMA Ch-19 SSI
  equations that a build agent had mis-reconstructed); ~640 reference methods verified 0-bug.
- **Ergonomics:** semantic aliases for reference methods (`_reference_common.register_semantic_aliases`)
  + consult round-budget fix; **smart method resolution + analysis-method aliases** in `dispatch.py`
  (`_METHOD_ALIASES`, selector-value directives, fuzzy did-you-mean) driven by the agent-suite triage
  (`module_work/module_feedback.json`). (A `fdm2d` wall-clock guard was added here too; `fdm2d`
  has since been removed in the Phase 1 consolidation — see `CONSOLIDATION_CHANGES.md`.)

**Open backlog (handed off, not done):** ~23 per-module param-name/value bugs (`module_feedback.json`);
module-*selection* mis-routing (e.g. Rankine guessed on the wrong module) and coverage gaps
(`retaining_walls` has no earth-pressure-coefficient method; `fem2d` no footing-SRM method).
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
- [ ] **P6 — (deferred) Text embeddings for recall.** Only if FTS5 lexical recall proves
      insufficient: text-embed the descriptions via `GenAIEngine.get_embedding()`. (Image
      embeddings remain rejected.)
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
