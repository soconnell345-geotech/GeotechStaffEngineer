# Funhouse Agent Expansion Plan

Comprehensive gap analysis of funhouse_agent capabilities vs the broader
GeotechStaffEngineer project. The funhouse_agent currently exposes **18 modules**
with **70 callable methods**. The project contains 13 additional agent modules
(wrapping external libraries) plus core modules that are not yet accessible.

> **Note:** Foundry wrappers (`foundry/`) are a separate Palantir deployment
> layer and are NOT part of this plan.

---

## Phase 1 — External Library Adapters (7 modules, ~20 methods)

High-value modules that follow the same flat-JSON → result-dict adapter pattern.
Each requires an optional external dependency with a `has_*()` guard.

| Priority | Module | Methods | External Dep | Capability |
|----------|--------|---------|-------------|------------|
| 1 | `opensees_agent` | pm4sand_cyclic_dss, bnwf_lateral_pile, site_response_1d | openseespy | Nonlinear dynamic FE: liquefaction DSS, advanced lateral pile, 1D site response |
| 2 | `pystrata_agent` | eql_site_response, linear_site_response | pystrata | Frequency-domain 1D SHAKE-type site response (Darendeli/Menq/custom curves) |
| 3 | `liquepy_agent` | cpt_liquefaction, field_correlations | liquepy | CPT-based liquefaction triggering (B&I 2014, LPI, LSN, LDI) — complements SPT-based seismic_geotech |
| 4 | `seismic_signals_agent` | response_spectrum, intensity_measures, rotd_spectrum, signal_processing | eqsig, pyrotd | Earthquake record processing, intensity characterization |
| 5 | `pyseismosoil_agent` | generate_curves, analyze_vs_profile | PySeismoSoil | MKZ/HH nonlinear soil curves + Vs30/f0/z1 — feeds into site response |
| 6 | `pystra_agent` | form, sorm, monte_carlo | pystra | FORM/SORM/Monte Carlo reliability analysis for probabilistic design |
| 7 | `salib_agent` | sobol_sample, sobol_analyze, morris_sample, morris_analyze | SALib | Global sensitivity analysis for parameter screening |

### Adapter pattern (same as existing 18):
- `METHOD_INFO` dict with parameter documentation
- `METHOD_REGISTRY` dict mapping method names → callables
- `has_*()` dependency check → clear error if library not installed
- Tests: mock-based, no API key or external library needed

---

## Phase 2 — File/Data Import Adapters (5 modules, ~15 methods)

These involve file parsing via the attachment mechanism.

| Module | Methods | External Dep | Capability |
|--------|---------|-------------|------------|
| `pygef_agent` | parse_cpt_file, parse_bore_file | pygef | CPT/borehole GEF and BRO-XML format parsing |
| `dxf_import` | discover_layers, parse_dxf_geometry, build_slope_geometry, build_fem_inputs | ezdxf | CAD geometry import (counterpart to existing dxf_export) |
| `pdf_import` | discover_pdf_content, extract_vector_geometry, extract_geometry_vision | PyMuPDF | Structured geometry extraction beyond current vision-only approach |
| `ags4_agent` | read_ags4, validate_ags4 | python-ags4 | UK/Commonwealth AGS4 geotechnical data format |
| `pydiggs_agent` | validate_diggs_schema, validate_diggs_dictionary | pydiggs | DIGGS 2.6 XML data interchange validation |

---

## Phase 3 — FEM High-Level APIs (1 module, 7 methods)

Expose fem2d's high-level analysis functions. These abstract away mesh setup
and provide reasonable defaults, making them usable via the ReAct agent pattern.

| Method | Capability |
|--------|------------|
| `analyze_slope_srm` | FEM slope stability via Strength Reduction Method (complements limit equilibrium) |
| `analyze_excavation` | Braced excavation with wall elements |
| `analyze_foundation` | Foundation stress-deformation analysis |
| `analyze_seepage` | Steady-state seepage (Laplace) |
| `analyze_consolidation` | Coupled Biot consolidation |
| `analyze_gravity` | Gravity loading of soil column |
| `analyze_staged` | Multi-phase staged construction |

No external dependency (numpy/scipy only).

---

## Phase 4 — Reference Agents & Visualization (optional, large scope)

### Reference lookup agents (geotech-references submodule):
- `dm7_agent` — 340+ NAVFAC DM7 functions
- `gec6/7/10/11/12/13` — FHWA design manual lookups
- `micropile_agent` — Micropile design reference
- `fema_p2192_agent` — Seismic design category
- `noaa_frost_agent` — Frost-protected shallow foundations
- `ufc_backfill/dewatering/expansive/pavement` — UFC series

### Subsurface visualization:
- `subsurface_characterization` — Plotly-based data visualization and trend analysis

### Other niche modules:
- `gstools_agent` — Geostatistical kriging and random fields
- `hvsrpy_agent` — HVSR ambient noise site characterization
- `swprocess_agent` — MASW surface wave dispersion
- `fdm2d` — 2D explicit FDM (FLAC-style) — fewer high-level APIs than fem2d

---

## Impact Summary

| Phase | Modules Added | Methods Added | Running Total | Key Capability |
|-------|--------------|---------------|---------------|----------------|
| Current | 18 | 70 | 70 | Core geotechnical analysis |
| Phase 1 | +7 | ~20 | ~90 | Advanced seismic, probabilistic design, sensitivity |
| Phase 2 | +5 | ~15 | ~105 | Data import (CPT, CAD, PDF), file parsing |
| Phase 3 | +1 | +7 | ~112 | FEM stress-deformation, SRM slope stability |
| Phase 4 | TBD | TBD | TBD | Standards reference, subsurface visualization |

---

## Notes

- **System prompt scaling**: Module catalog table grows but stays manageable.
  The LLM uses `list_methods` → `describe_method` for discovery.
- **Calc package**: Currently supports 13 of 18 modules. New modules won't have
  calc package support initially — that's a separate effort.
- **Test coverage**: Each adapter needs mock-based tests following
  `funhouse_agent/tests/` patterns.
