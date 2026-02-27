# Subsurface Characterization Design

## Purpose

Visualize and characterize subsurface investigation data using interactive Plotly
figures exported as self-contained HTML. Supports delivery via Databricks
`displayHTML()`, SharePoint iframes, Foundry agent responses, or saved HTML files.

## Architecture

### Data Model

Internal data model that all input formats populate:

```
SiteModel
  └── Investigation[]
        ├── LithologyInterval[]   (interpreted soil/rock layers)
        └── PointMeasurement[]    (discrete measurements at depth)
```

- **PointMeasurement**: NOT layer-averaged. Single test at a single depth.
- **LithologyInterval**: Interpreted from field log. USCS classification attached.
- **Investigation**: Boring, CPT, test pit, monitoring well.
- **SiteModel**: Top-level container for entire project.

### Input Formats

All populate the same SiteModel:

| Format | Loader | Notes |
|--------|--------|-------|
| DIGGS 2.6/2.5.a XML | `parse_diggs()` | Custom `xml.etree` parser (pydiggs is validation-only) |
| Nested dict | `load_site_from_dict()` | Direct Python dict structure |
| CSV files | `load_site_from_csv()` | Borings + measurements + optional lithology |
| pygef CPTParseResult | `load_cpt_to_investigation()` | Bridge from `pygef_agent` |

### Visualizations

All return `PlotResult` wrapping a `go.Figure`:

| Plot | Function | X-axis | Y-axis |
|------|----------|--------|--------|
| Parameter vs depth | `plot_parameter_vs_depth()` | Value | Depth/elevation |
| Atterberg limits | `plot_atterberg_limits()` | Moisture % | Depth/elevation |
| Multi-parameter | `plot_multi_parameter()` | Values | Depth (shared) |
| Plan view | `plot_plan_view()` | Easting | Northing |
| Cross-section | `plot_cross_section()` | Distance | Elevation/depth |

## Units and Conventions

All SI, consistent with rest of codebase:

| Quantity | Unit | Notes |
|----------|------|-------|
| Depth | m | Positive downward from ground surface |
| Elevation | m | From vertical datum |
| Pressure/stress | kPa | |
| Unit weight | kN/m³ | |
| Angles | degrees | |
| SPT | blows/0.3m | N_spt (raw), N60 (corrected) |
| CPT | kPa | qc, fs, u2 all in kPa |
| Moisture | % | wn, LL, PL as percentages |

## Standard Parameter Names

Used consistently across data model, DIGGS parser, CSV loader:

```
N_spt, N60, N160, qc_kPa, fs_kPa, u2_kPa, Rf_pct,
cu_kPa, phi_deg, qu_kPa, RQD_pct, gamma_kNm3,
wn_pct, LL_pct, PL_pct, PI_pct,
e0, Cc, Cr, sigma_p_kPa, Su_vane_kPa, pocket_pen_kPa
```

## Statistics

Reference: Phoon & Kulhawy (1999), Canadian Geotechnical Journal.

- **Linear trend**: `value = slope * depth + intercept`
- **Log-linear trend**: `ln(value) = slope * ln(depth) + intercept`
- **COV**: coefficient of variation = std_residual / mean_value
- **Grouped trends**: separate regression per USCS class or per investigation

## DIGGS Parser

Custom parser using `xml.etree.ElementTree` (stdlib only):

- Auto-detects 2.6 vs 2.5.a namespace from root element
- Resolves `xlink:href="#gml_id"` references to associate tests with borings
- Extracts: Project name, Borehole coords/depth, LithologyObservation,
  DrivenPenetrationTest (SPT), AtterbergLimitsTest, MoistureContent,
  WaterLevelObservation

## Cross-Section Design Decision

Phase 1: **NO interpreted connections between borings**. Only raw data columns.
This avoids subjective layer correlation decisions. Future phases may add
interpreted subsurface profiles.

## Edge Cases

- Empty site model: plots return valid but empty figures
- Missing USCS: classified as "Unknown" with gray color
- Missing coordinates: default to (0, 0)
- Single investigation in cross-section: returns minimal plot
- Parameter with no data: returns empty PlotResult
- log-linear with zero values: masked out (log(0) undefined)

## References

- Phoon, K.K. & Kulhawy, F.H. (1999). "Characterization of geotechnical
  variability." Canadian Geotechnical Journal, 36(4), 612-624.
- DIGGS 2.6 Schema: http://diggsml.org/schemas/2.6
- ASTM D2487 (USCS classification)
