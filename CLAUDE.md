# GeotechStaffEngineer

Python toolkit for LLM-based geotechnical engineering agents.
17 analysis modules + groundhog wrapper + OpenSees agent + DM7 equations.

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

## Module Inventory (1100 tests)

| Module | Tests | Purpose |
|--------|-------|---------|
| bearing_capacity | 45 | Shallow foundations (CBEAR/Vesic/Meyerhof) |
| settlement | 39 | Consolidation & immediate (CSETT) |
| axial_pile | 55 | Driven pile capacity (Nordlund/Tomlinson/Beta) |
| sheet_pile | 26 | Cantilever/anchored walls (Rankine/Coulomb) |
| lateral_pile | 58 | Lateral pile (COM624P, 7 p-y models, FD solver) |
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

Other components: groundhog_agent (90 methods), DM7Eqs (382 functions, 2008 tests)

## Working on a Module

1. Read the module's `DESIGN.md` first for theory and conventions
2. Read `__init__.py` for the public API
3. Run that module's tests: `pytest module_name/ -v`
4. Full regression: `pytest bearing_capacity/ settlement/ axial_pile/ sheet_pile/ lateral_pile/validation.py pile_group/ wave_equation/ geotech_common/ drilled_shaft/ seismic_geotech/ retaining_walls/ ground_improvement/ slope_stability/ downdrag/ opensees_agent/ -q`

## Environment

- Windows 11, Python 3.14.3, venv at `.venv/`
- Git repo: github.com/soconnell345-geotech/GeotechStaffEngineer (private)
- numpy >=2.0: use `np.trapezoid` (was `np.trapz`)
