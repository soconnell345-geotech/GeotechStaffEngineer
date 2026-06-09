# GeotechStaffEngineer

Python toolkit for LLM-based geotechnical engineering agents. 29 analysis modules covering foundations, piles, slopes, seismic analysis, ground improvement, FEM, and more.

## Installation

```bash
# Core package (numpy + scipy + geotech-references)
pip install geotech-staff-engineer

# With all optional agent libraries
pip install geotech-staff-engineer[full]

# Or install individual extras
pip install geotech-staff-engineer[plot,groundhog,opensees]
```

## Quick Start

```python
from bearing_capacity import Footing, SoilLayer, BearingSoilProfile, BearingCapacityAnalysis

footing = Footing(width=2.0, length=10.0, depth=1.5, shape="strip")
layer = SoilLayer(friction_angle=30.0, cohesion=10.0, unit_weight=18.0, thickness=10.0)
profile = BearingSoilProfile(layer1=layer, gwt_depth=5.0)

analysis = BearingCapacityAnalysis(footing=footing, soil=profile)
result = analysis.compute()
print(result.summary())
```

## Modules

All units are SI (meters, kPa, kN, degrees). Every module returns dataclasses with `.summary()` and `.to_dict()` methods for easy LLM integration.

### Core Analysis (19 modules)

| Module | Purpose |
|--------|---------|
| `bearing_capacity` | Shallow foundations — Vesic, Meyerhof, Hansen |
| `settlement` | Consolidation and immediate settlement |
| `axial_pile` | Driven pile capacity — Nordlund, Tomlinson, Beta |
| `sheet_pile` | Cantilever and anchored sheet pile walls |
| `soe` | Support of excavation — braced/cantilever walls, stability, ground anchors |
| `lateral_pile` | Lateral pile analysis — COM624P, 8 p-y models |
| `pile_group` | Rigid-cap pile groups — 6-DOF, efficiency factors |
| `wave_equation` | Smith 1-D wave equation — bearing graph, drivability |
| `drilled_shaft` | Drilled shaft capacity — GEC-10 alpha/beta/rock socket |
| `seismic_geotech` | Site classification, M-O pressures, SPT liquefaction (NCEER/Youd-2001) |
| `retaining_walls` | Cantilever and MSE retaining walls |
| `ground_improvement` | Aggregate piers, wick drains, vibro-compaction |
| `slope_stability` | Fellenius, Bishop, Spencer — circular slip, soil nails |
| `downdrag` | Neutral plane method, dragload estimation |
| `geotech_common` | Shared soil profile, adapters, plotting utilities |
| `calc_package` | Calculation package report generation |
| `subsurface_characterization` | Subsurface data I/O: DIGGS parser + Plotly visualizations + trend stats, plus optional format adapters for GEF/BRO-XML CPT/borehole (pygef), AGS4 (python-ags4), and DIGGS schema/dictionary validation (pydiggs) |
| `dxf_import` | DXF CAD import for slope stability geometry |
| `fem2d` | 2D plane-strain FEM — CST/Q4/beam, MC/HS, SRM, seepage, consolidation |

### Library Wrapper Agents (10 modules)

Each agent wraps a third-party geotechnical library with a dict-based API for LLM tool use.

| Module | Library | Purpose |
|--------|---------|---------|
| `opensees_agent` | OpenSeesPy | PM4Sand cyclic DSS, 1D site response |
| `pystrata_agent` | pystrata | 1D equivalent-linear site response |
| `seismic_signals_agent` | eqsig + pyrotd | Earthquake signal processing |
| `liquepy_agent` | liquepy | Boulanger & Idriss (2014) liquefaction triggering — CPT (LPI/LSN/LDI) and SPT |
| `hvsrpy_agent` | hvsrpy | HVSR site characterization |
| `gstools_agent` | gstools | Geostatistical kriging and random fields |
| `salib_agent` | SALib | Sobol and Morris sensitivity analysis |
| `swprocess_agent` | swprocess | MASW surface wave dispersion |
| `pystra_agent` | pystra | FORM/SORM/Monte Carlo reliability |
| `groundhog_agent` | groundhog | Site investigation and soil mechanics |

> The former `pygef_agent` (pygef), `ags4_agent` (python-ags4), and `pydiggs_agent`
> (pydiggs) library wrappers were folded into `subsurface_characterization` as
> optional, dependency-backed format adapters — one module now covers
> ingest + validate + visualize across DIGGS, GEF/BRO-XML, and AGS4. Their optional
> dependencies remain installable (see Optional Extras).

## Optional Extras

| Extra | Libraries |
|-------|-----------|
| `plot` | matplotlib |
| `calc` | jinja2 |
| `groundhog` | groundhog |
| `opensees` | openseespy |
| `pystrata` | pystrata |
| `seismic-signals` | eqsig, pyrotd |
| `liquepy` | liquepy |
| `hvsrpy` | hvsrpy |
| `gstools` | gstools |
| `salib` | SALib |
| `swprocess` | swprocess |
| `pystra` | pystra |
| `subsurface` | pygef, python-ags4, pydiggs (subsurface_characterization format adapters) |
| `pygef` | pygef (alias) |
| `ags4` | python-ags4 (alias) |
| `pydiggs` | pydiggs (alias) |
| `dxf` | ezdxf |
| `full` | All of the above |

## Unified Liquefaction

Liquefaction triggering is exposed to the agent through a single `liquefaction`
tool that auto-routes by input type and method:

- **CPT** input (cone resistance `q_c` / sleeve friction `f_s`) → Boulanger &
  Idriss (2014) CPT procedure via `liquepy`, with LPI / LSN / LDI indices.
- **SPT** input (`N160` blow counts) → Boulanger & Idriss (2014) by default
  (`method="bi2014"`), or the legacy NCEER / Youd et al. (2001) simplified
  procedure via `method="nceer2001"` for code-compliance work that cites it.

B&I-2014 is the default for both. The underlying per-module functions remain
available directly: `liquepy_agent.analyze_cpt_liquefaction` /
`analyze_spt_liquefaction` (B&I-2014) and `seismic_geotech.evaluate_liquefaction`
(NCEER/Youd-2001 SPT). The SPT B&I-2014 path is composed from `liquepy`'s tested
B&I-2014 building blocks — `liquepy` ships no packaged SPT triggering object.

## Related Package

[geotech-references](https://pypi.org/project/geotech-references/) — Digitized NAVFAC DM7 and FHWA GEC reference library (installed automatically as a dependency).

## License

MIT
