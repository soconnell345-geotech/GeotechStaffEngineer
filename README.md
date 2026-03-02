# GeotechStaffEngineer

Python toolkit for LLM-based geotechnical engineering agents. 35 analysis modules covering foundations, piles, slopes, seismic analysis, ground improvement, FEM, and more.

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

### Core Analysis (20 modules)

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
| `seismic_geotech` | Site classification, M-O pressures, liquefaction |
| `retaining_walls` | Cantilever and MSE retaining walls |
| `ground_improvement` | Aggregate piers, wick drains, vibro-compaction |
| `slope_stability` | Fellenius, Bishop, Spencer — circular slip, soil nails |
| `downdrag` | Neutral plane method, dragload estimation |
| `geotech_common` | Shared soil profile, adapters, plotting utilities |
| `calc_package` | Calculation package report generation |
| `subsurface_characterization` | DIGGS parser, Plotly visualizations, trend stats |
| `wind_loads` | ASCE 7-22 wind on freestanding walls and fences |
| `dxf_import` | DXF CAD import for slope stability geometry |
| `fem2d` | 2D plane-strain FEM — CST/Q4/beam, MC/HS, SRM, seepage, consolidation |

### Library Wrapper Agents (15 modules)

Each agent wraps a third-party geotechnical library with a dict-based API for LLM tool use.

| Module | Library | Purpose |
|--------|---------|---------|
| `opensees_agent` | OpenSeesPy | PM4Sand cyclic DSS, BNWF pile, 1D site response |
| `pystrata_agent` | pystrata | 1D equivalent-linear site response |
| `seismic_signals_agent` | eqsig + pyrotd | Earthquake signal processing |
| `liquepy_agent` | liquepy | CPT-based liquefaction triggering |
| `pygef_agent` | pygef | GEF/BRO-XML CPT and borehole parsing |
| `hvsrpy_agent` | hvsrpy | HVSR site characterization |
| `gstools_agent` | gstools | Geostatistical kriging and random fields |
| `ags4_agent` | python-ags4 | AGS4 data format reading and validation |
| `salib_agent` | SALib | Sobol and Morris sensitivity analysis |
| `pyseismosoil_agent` | PySeismoSoil | Nonlinear soil curve calibration |
| `swprocess_agent` | swprocess | MASW surface wave dispersion |
| `geolysis_agent` | geolysis | Soil classification and SPT corrections |
| `pystra_agent` | pystra | FORM/SORM/Monte Carlo reliability |
| `pydiggs_agent` | pydiggs | DIGGS 2.6 XML validation |
| `groundhog_agent` | groundhog | Site investigation and soil mechanics |

## GUIs

Interactive browser-based GUIs built with Plotly Dash:

| GUI | Command | Purpose |
|-----|---------|---------|
| `slope_stability_gui.py` | `python slope_stability_gui.py` | Slope stability analysis with live preview |
| `fem2d_gui.py` | `python fem2d_gui.py` | 2D FEM analysis — gravity, foundation, SRM, excavation, seepage, consolidation |

Requires `pip install geotech-staff-engineer[gui]` for Dash and Plotly dependencies.

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
| `pygef` | pygef |
| `hvsrpy` | hvsrpy |
| `gstools` | gstools |
| `ags4` | python-ags4 |
| `salib` | SALib |
| `pyseismosoil` | PySeismoSoil |
| `swprocess` | swprocess |
| `geolysis` | geolysis |
| `pystra` | pystra |
| `pydiggs` | pydiggs |
| `dxf` | ezdxf |
| `gui` | dash, plotly |
| `full` | All of the above |

## Related Package

[geotech-references](https://pypi.org/project/geotech-references/) — Digitized NAVFAC DM7 and FHWA GEC reference library (installed automatically as a dependency).

## License

MIT
