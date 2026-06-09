# Funhouse Agent Expansion Plan — COMPLETE

All four expansion phases are **complete**. The funhouse_agent now exposes
**46 modules** with **~890 callable methods**, up from the original 18 modules / 70 methods.

> **Phase 1 consolidation (2026-06-09):** `pyseismosoil`, `fdm2d`, `geolysis`,
> and `wind_loads` were subsequently removed as redundant/out-of-scope. The phase
> notes below are kept as the historical record of the original build-out.

> **Note:** Foundry wrappers (`foundry/`) are a separate Palantir deployment
> layer and are NOT part of the funhouse_agent.

---

## Phase 1 — External Library Adapters ✅

6 modules added: `opensees`, `pystrata`, `liquepy`, `seismic_signals`,
`pystra`, `salib`. (`pyseismosoil` was added here originally, later removed.)

Each follows the standard adapter pattern with `METHOD_REGISTRY` / `METHOD_INFO`,
optional dependency guards, and mock-based tests.

---

## Phase 2 — File/Data Import Adapters ✅

5 modules added: `pygef`, `dxf_import`, `pdf_import`, `ags4`, `pydiggs`.

File parsing via attachment mechanism or direct path input.

---

## Phase 3 — FEM High-Level APIs ✅

2 modules added: `fem2d`, `subsurface`. (`fdm2d` was added here originally, later removed.)

Exposes high-level analysis functions (gravity, foundation, slope SRM,
excavation, seepage, consolidation, staged construction) and subsurface
data visualization.

---

## Phase 4 — Reference Agents & Remaining Modules ✅

18 modules added:
- **14 geotech-references agents**: `dm7` (340+ methods), `gec6`, `gec7`,
  `gec10`, `gec11`, `gec12`, `gec13`, `micropile`, `fema_p2192`, `noaa_frost`,
  `ufc_backfill`, `ufc_dewatering`, `ufc_expansive`, `ufc_pavement`
- **4 additional modules**: `gstools`, `hvsrpy`, `swprocess`, `calc_package`

Reference adapters use a shared factory (`_reference_common.py`) that
auto-discovers functions via `inspect.getmembers()`. GEC/micropile adapters
include text retrieval methods (retrieve_section, search_sections, list_chapters,
load_chapter). DM7 handles name collisions across 15 chapter modules with
chapter-key prefixes.

---

## Final Totals

| Phase | Modules | Methods | Key Capability |
|-------|---------|---------|----------------|
| Original | 18 | 70 | Core geotechnical analysis |
| Phase 1 | +6 | +~45 | Advanced seismic, probabilistic, sensitivity |
| Phase 2 | +5 | +~30 | Data import (CPT, CAD, PDF, AGS4, DIGGS) |
| Phase 3 | +2 | +~35 | FEM, subsurface visualization |
| Phase 4 | +17 | +~711 | Standards references (DM7/GEC/UFC/FEMA/NOAA), geostatistics, HVSR, MASW |
| Consolidation | −2 | — | Removed `geolysis`, `wind_loads` (redundant/out-of-scope) |
| **Total** | **46** | **~890** | **Full geotechnical toolkit** |

---

## Test Coverage

- 106 core agent/engine/notebook tests
- 149 analysis adapter tests (Phases 1–3)
- 163 reference adapter tests (Phase 4)
- **418 total funhouse_agent tests** (all passing)

---

## Remaining Gaps (Not Blocking)

- **Calc package**: Supports 13 of the original 18 analysis modules. The 32 new
  modules do not have calc package (HTML/PDF report) support — separate effort.
- **System prompt**: Auto-generated from MODULE_REGISTRY; already reflects all
  46 modules via `list_methods` → `describe_method` discovery.
