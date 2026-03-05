# GeotechStaffEngineer

Python toolkit for LLM-based geotechnical engineering agents.
35 analysis modules + groundhog wrapper + OpenSees agent + pyStrata agent + seismic signals agent + liquepy agent + pygef agent + hvsrpy agent + GSTools agent + AGS4 agent + SALib agent + PySeismoSoil agent + swprocess agent + geolysis agent + pystra agent + pydiggs agent + subsurface characterization + wind loads (ASCE 7-22) + DXF import + PDF import + fem2d (2D plane-strain FEM with staged construction) + DM7 equations + trial_agent (Claude API tool_use integration) + chat_agent (ReAct agent for text-only chat functions) + funhouse_agent (engine-agnostic agent with vision).

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

## Module Inventory (2792 module + 142 harness + 3299 ref = 6233 tests)

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
| geotech_common | 284 | SoilProfile (82) + checks (93) + adapters (89) + plots (21) |
| opensees_agent | 106 | PM4Sand cyclic DSS, BNWF lateral pile, 1D site response |
| pystrata_agent | ~57 | 1D EQL site response (SHAKE-type, Darendeli/Menq/custom) |
| seismic_signals | 74 | Earthquake signal processing (eqsig/pyrotd) |
| liquepy_agent | 59 | CPT-based liquefaction triggering (B&I 2014, LPI, LSN, LDI) |
| pygef_agent | 45 | CPT/borehole file parser (GEF/BRO-XML) |
| hvsrpy_agent | 37 | HVSR site characterization from ambient noise |
| gstools_agent | 69 | Geostatistical kriging, variogram fitting, random fields |
| ags4_agent | 39 | AGS4 geotechnical data format reader/validator |
| salib_agent | 35 | Sobol & Morris sensitivity analysis |
| pyseismosoil_agent | 41 | MKZ/HH nonlinear soil curve calibration + Vs profiles |
| swprocess_agent | 30 | MASW surface wave dispersion analysis |
| geolysis_agent | 65 | USCS/AASHTO soil classification + SPT corrections + bearing capacity |
| pystra_agent | 43 | FORM/SORM/Monte Carlo structural reliability analysis |
| pydiggs_agent | 31 | DIGGS 2.6 XML schema and dictionary validation |
| subsurface_characterization | 145 | Subsurface data visualization (DIGGS parser w/ 20 test types, Plotly plots, trend stats) |
| wind_loads | 62 | ASCE 7-22 wind loads on freestanding walls and fences (Ch 29.3) |
| dxf_import | 97 | DXF CAD import for slope stability + FEM (discover layers, parse geometry, build SlopeGeometry/FEM inputs) |
| pdf_import | 56 | PDF cross-section import (PyMuPDF vector extraction, LLM vision extraction, geometry conversion) |
| fem2d | 271 | 2D plane-strain FEM (CST/Q4/beam, MC/HS, SRM, excavation, pore pressures, seepage, consolidation, staged construction) |

Other components: groundhog_agent (90 methods), geotech-references submodule (382 DM7 + 95 GEC/micropile + 10 FEMA + 9 NOAA + 35 UFC functions, 3299 tests), foundry_test_harness (142 tests), trial_agent (100 tests), chat_agent (42 tests), funhouse_agent (42 tests)

## GUIs

### Plotly Dash (browser-based)
| GUI | Port | Purpose |
|-----|------|---------|
| `slope_stability_gui.py` | 8051 | Slope stability: geometry, soil layers, GWT, Bishop/Spencer grid search |
| `fem2d_gui.py` | 8055 | 2D FEM: gravity, foundation, slope SRM, excavation, seepage, consolidation |

Run: `python slope_stability_gui.py` or `python fem2d_gui.py` — opens browser automatically.

### Qt Desktop (PyQt5/PySide2)
| GUI | Purpose |
|-----|---------|
| `geotech_qt_gui.py` | Main app: Bearing Capacity, Settlement, FEM 2D tabs |
| `slope_stability_qt.py` | Standalone slope stability: circular/noncircular, entry/exit limits, slice visualization, contact stress plot |
| `fem2d_qt.py` | Standalone FEM 2D: tabbed inputs, mesh preview, BC controls, FOS vs displacement |

Run: `python geotech_qt_gui.py` or `python slope_stability_qt.py` or `python fem2d_qt.py`

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

## Trial Agent (Claude API Integration)

`trial_agent/` uses the Claude API `tool_use` feature to give Claude direct access to all Foundry agents. The agent receives 3 meta-tools:

- `call_agent(agent, method, params)` — Route to any Foundry agent function
- `list_methods(agent, category)` — Discover available methods for an agent
- `describe_method(agent, method)` — Get parameter details and usage info

| File | Purpose |
|------|---------|
| `__init__.py` | Package init |
| `__main__.py` | Entry point for `python -m trial_agent` |
| `cli.py` | CLI interface for interactive queries |
| `system_prompt.py` | **Primary agent prompt** — the canonical system prompt (dynamic agent count) |
| `agent_registry.py` | Maps agent names to Foundry functions |
| `tools.py` | Tool definitions for Claude API |
| `test_run.py` | 10-question trial run |
| `test_run_30.py` | 30-question trial run |
| `test_run_60.py` | 60-question trial run |
| `test_run_100.py` | 100-question comprehensive trial |
| `analyze_descriptions.py` | Utility to analyze method descriptions |

Note: Trial agent tests require a valid `ANTHROPIC_API_KEY` and incur API costs. They are not included in the standard regression command.

## Chat Agent (Text-Only LLM Integration)

`chat_agent/` is a ReAct agent that gives any text-in/text-out chat function access to all Foundry agents via `<tool_call>` tag parsing. Designed for Databricks notebooks using `fh_prompter.chat()`. No new dependencies — reuses `trial_agent/agent_registry.py` and `trial_agent/system_prompt.py`.

| File | Purpose |
|------|---------|
| `__init__.py` | Exports: `GeotechChatAgent`, `AgentResult`, `parse_response`, `ToolCall`, `build_system_prompt` |
| `parser.py` | Parse `<tool_call>JSON</tool_call>` tags from LLM text responses |
| `react_prompt.py` | ReAct protocol instructions + canonical system prompt |
| `agent.py` | `GeotechChatAgent` class (ReAct loop), `ConversationHistory`, `dispatch_tool()` |
| `tests/test_chat_agent.py` | 42 tests (all use mock chat functions, no API key needed) |

Usage:
```python
from chat_agent import GeotechChatAgent

agent = GeotechChatAgent(chat_fn=fh_prompter.chat, verbose=True)
result = agent.ask("Calculate bearing capacity of a 2m footing, phi=30")
print(result.answer)

# Follow-up (sees prior context)
result2 = agent.ask("Now estimate settlement")

# Start fresh
agent.reset()
```

Run: `pytest chat_agent/ -v`

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

`funhouse_agent/` provides an engine-agnostic geotechnical agent with text + vision capabilities. Works with any AI backend satisfying the `GenAIEngine` protocol. Self-contained dispatch layer routes tool calls directly to 16 analysis modules via internal adapters — no dependency on `foundry/` files.

| File | Purpose |
|------|---------|
| `__init__.py` | Exports: `GeotechAgent`, `GenAIEngine`, `ClaudeEngine`, `AgentResult` |
| `engine.py` | `GenAIEngine` Protocol + `ClaudeEngine` adapter |
| `agent.py` | `GeotechAgent` class (ReAct loop + vision dispatch) |
| `dispatch.py` | Tool dispatch — routes to adapters (not foundry) |
| `system_prompt.py` | Self-contained system prompt (16 modules) |
| `vision_tools.py` | Vision tool definitions and dispatch |
| `adapters/` | 16 adapter modules bridging flat JSON → analysis module APIs |
| `tests/` | 42 tests (mock engines, no API key needed) |

Usage:
```python
from funhouse_agent import GeotechAgent, ClaudeEngine

# With PrompterAPI (Databricks) — works natively
agent = GeotechAgent(genai_engine=prompter_api)

# With Claude
agent = GeotechAgent(genai_engine=ClaudeEngine())

result = agent.ask("Calculate bearing capacity of 2m footing, phi=30")
```

Run: `pytest funhouse_agent/ -v`

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
