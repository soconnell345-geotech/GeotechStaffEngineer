# GeotechStaffEngineer

Python toolkit for LLM-based geotechnical engineering agents.
30 analysis modules + groundhog wrapper + OpenSees agent + pyStrata agent + seismic signals agent + liquepy agent + pygef agent + hvsrpy agent + GSTools agent + AGS4 agent + SALib agent + PySeismoSoil agent + swprocess agent + geolysis agent + pystra agent + pydiggs agent + DM7 equations + trial_agent (Claude API tool_use integration).

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
- **Foundry wrappers** (`*_agent_foundry.py` in root): 3 functions each (agent/list/describe)

## Module Inventory (1735 module + 121 harness + 2810 ref = 4666 tests)

| Module | Tests | Purpose |
|--------|-------|---------|
| bearing_capacity | 45 | Shallow foundations (CBEAR/Vesic/Meyerhof) |
| settlement | 39 | Consolidation & immediate (CSETT) |
| axial_pile | 55 | Driven pile capacity (Nordlund/Tomlinson/Beta) |
| sheet_pile | 26 | Cantilever/anchored walls (Rankine/Coulomb) |
| lateral_pile | 66 | Lateral pile (COM624P, 8 p-y models, FD solver) |
| pile_group | 72 | Rigid cap groups (6-DOF, Converse-Labarre) |
| wave_equation | 45 | Smith 1-D wave equation (bearing graph, drivability) |
| drilled_shaft | 48 | GEC-10 alpha/beta/rock socket |
| seismic_geotech | 71 | Site class, M-O pressures, liquefaction |
| retaining_walls | 70 | Cantilever + MSE walls (GEC-11) |
| ground_improvement | 43 | Aggregate piers, wick drains, surcharge, vibro (GEC-13) |
| slope_stability | 55 | Fellenius/Bishop/Spencer, circular slip, grid search |
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

Other components: groundhog_agent (90 methods), geotech-references submodule (382 DM7 + 95 GEC/micropile + 10 FEMA + 9 NOAA functions, 3028 tests), foundry_test_harness (121 tests), trial_agent (100 tests)

## Foundry Test Harness

`foundry_test_harness/` validates the 30+ Foundry agent functions via JSON-in/JSON-out:

| File | Tests | Purpose |
|------|-------|---------|
| test_tier1_textbook.py | 58 | Individual functions vs textbook/published answers |
| test_tier2_workflows.py | 14 | Multi-function engineering workflows (e.g., classify → SPT → bearing → settlement) |
| test_tier3_crosscheck.py | 10 | Cross-agent consistency (same problem, different agents) |
| test_tier4_error_handling.py | 39 | Bad JSON, unknown methods, missing params, invalid values |

Supporting files: `harness.py` (FoundryAgentHarness class), `scenarios.py` (reusable problem definitions)

Run: `pytest foundry_test_harness/ -v`

## Trial Agent (Claude API Integration)

`trial_agent/` uses the Claude API `tool_use` feature to give Claude direct access to all 30 Foundry agents. The agent receives 3 meta-tools:

- `call_agent(agent, method, params)` — Route to any of the 30 Foundry agent functions
- `list_methods(agent, category)` — Discover available methods for an agent
- `describe_method(agent, method)` — Get parameter details and usage info

| File | Purpose |
|------|---------|
| `__init__.py` | Package init |
| `__main__.py` | Entry point for `python -m trial_agent` |
| `cli.py` | CLI interface for interactive queries |
| `system_prompt.py` | Engineering system prompt for Claude |
| `agent_registry.py` | Maps agent names to Foundry functions |
| `tools.py` | Tool definitions for Claude API |
| `test_run.py` | 10-question trial run |
| `test_run_30.py` | 30-question trial run |
| `test_run_60.py` | 60-question trial run |
| `test_run_100.py` | 100-question comprehensive trial |
| `analyze_descriptions.py` | Utility to analyze method descriptions |

Note: Trial agent tests require a valid `ANTHROPIC_API_KEY` and incur API costs. They are not included in the standard regression command.

## Working on a Module

1. Read the module's `DESIGN.md` first for theory and conventions
2. Read `__init__.py` for the public API
3. Run that module's tests: `pytest module_name/ -v`
4. Full regression: `pytest bearing_capacity/ settlement/ axial_pile/ sheet_pile/ lateral_pile/validation.py pile_group/ wave_equation/ geotech_common/ drilled_shaft/ seismic_geotech/ retaining_walls/ ground_improvement/ slope_stability/ downdrag/ opensees_agent/ pystrata_agent/ seismic_signals_agent/ liquepy_agent/ pygef_agent/ hvsrpy_agent/ gstools_agent/ ags4_agent/ salib_agent/ pyseismosoil_agent/ swprocess_agent/ geolysis_agent/ pystra_agent/ pydiggs_agent/ foundry_test_harness/ geotech-references/tests/ -q`

## Environment

- Windows 11, Python 3.14.3, venv at `.venv/`
- Git repo: github.com/soconnell345-geotech/GeotechStaffEngineer (private)
- Git submodule: `geotech-references/` → github.com/soconnell345-geotech/geotech-references (DM7 + future GEC refs)
- numpy >=2.0: use `np.trapezoid` (was `np.trapz`)
