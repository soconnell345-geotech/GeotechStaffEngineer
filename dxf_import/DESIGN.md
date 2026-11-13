# DXF Import Module — Design Notes

## Purpose

Import CAD cross-sections (DXF format from AutoCAD / Civil 3D) and translate
them into `SlopeGeometry` for slope stability analysis. DXF provides geometry
only — soil properties must come from the user.

## Three-Step Workflow

1. **`discover_layers(filepath)`** — Read DXF, list all layers with entity
   counts, types, sample text, and bounding boxes.
2. **`parse_dxf_geometry(filepath, layer_mapping, units)`** — Extract
   coordinates from user-mapped layers, convert units (checked against the
   drawing's own header), produce sorted profiles.
3. **`build_slope_geometry(parse_result, soil_properties)`** — Assemble
   `SlopeGeometry` with user-supplied soil parameters.

## DXF Entity Mapping

| Entity Type | Used For | Notes |
|-------------|----------|-------|
| LWPOLYLINE | Surface, boundaries, GWT | Primary geometry. Vertices extracted. |
| POLYLINE | Surface, boundaries, GWT | 3D polylines also supported. |
| LINE | Surface (merged), boundaries, **nails** | Individual segments merged for profiles. Each nail = one LINE from head to tip. |
| SPLINE | Surface, boundaries | Flattened to polyline via `ezdxf.flattening()`. |
| TEXT / MTEXT | Annotations | Insertion point (in metres) + text content extracted. A mapped annotation layer that is not in the drawing is named in warnings. |

## Layer Name Convention

DXF layer names are arbitrary — every CAD operator names them differently.
That's why we use a discover-then-parse workflow: the user inspects layer
names and creates a `LayerMapping`.

## Unit Handling

Nothing is assumed about the units a drawing is in (GM-327, 2026-11-13;
`units` used to default to metres and was never compared with the drawing,
so a drawing in mm came out 1000 times too large).

`parse_dxf_geometry(units=None, ignore_header_units=False)` sets the caller's
`units` beside the drawing's own `$INSUNITS` header (the value
`discover_layers` reports as `units_hint`):

| `units` | Header | Result |
|---|---|---|
| left out | states m, mm, cm, ft or in | the header's units are used; `units_source` says so |
| left out | unitless, or a unit not converted (km, miles, ...) | refused: give `units` |
| given | unitless | the given units |
| given | the same | the given units |
| given | different | refused, naming both; with `ignore_header_units=True` the given units are used and a warning names both |

- A header left at a template's value is common (a drawing in feet whose
  header says inches), which is why the caller can overrule it, and why
  doing so takes an explicit input.
- The result carries `units_used`, `units_source` and `header_units`.
- All output is SI metres: surface, boundaries, water table, nails and the
  insertion points of text annotations (which were copied in drawing units
  until GM-327).
- Supported: m, mm, cm, ft, in

## Soil and nail properties (the agent adapter)

A drawing gives geometry only. `funhouse_agent/adapters/dxf_import_adapter.py`
(`soil_assignments`, shared with the PDF import adapter) therefore assumes
no unit weight, strength or stiffness:

| Tool | Refused when left out | Stands in when left out, and listed in `defaults_used` |
|---|---|---|
| `build_slope_geometry` | `gamma`; `phi` (drained layer); `cu` (undrained layer) | `c_prime` 0, `analysis_mode` 'drained', `gamma_sat` = the layer's `gamma` |
| `build_fem_inputs` | `gamma`, `E`, `nu`; `phi` (model mc or hs) | `c` 0, `psi` 0, `model` 'mc' |

- `gamma_sat` left out takes the layer's own `gamma` (`SlopeSoilLayer`'s
  rule: one unit weight above and below the water table); the output
  carries it, so it is listed as the value it took.
- `defaults_used` keys are `soil_properties[<name>].<input>` and
  `nail_defaults.<input>`. The key is absent when nothing stood in.
- Nails: a drawing gives each nail's head, length and inclination. The bar
  diameter (25 mm), drill-hole diameter (150 mm), fy (420 MPa), grout-ground
  bond stress (100 kPa) and horizontal spacing (1.5 m) that
  `build_slope_geometry` fills in are `slope_stability.nails.SoilNail`'s own
  defaults (`converter.NAIL_PROPERTY_DEFAULTS`); each one not given in
  `nail_defaults` is listed. The bond stress depends on the ground and
  should be given.
- The dataclasses `SoilPropertyAssignment` and `FEMSoilPropertyAssignment`
  still carry field defaults (gamma 18, E 30000, nu 0.3, ...) for a library
  caller; the agent's route does not use them.

## Nail Convention

- Each LINE on the nail layer = one nail
- Head = leftmost endpoint (closer to slope face)
- Tip = rightmost endpoint (deeper into slope)
- Inclination computed from head/tip geometry (degrees below horizontal)
- User supplies nail properties (bar diameter, bond stress, etc.) via `nail_defaults` dict; see "Soil and nail properties" for what stands in
- Only LINE entities are read as nails; anything else on the nail layer is counted in the parse result's warnings

## Layer Stacking

For multi-layer slopes:
- Surface profile = top of layer 1
- Boundary 1 = bottom of layer 1 / top of layer 2
- Boundary 2 = bottom of layer 2 / top of layer 3
- etc.

Layer elevations: `top_elevation = max(z)` along top profile,
`bottom_elevation` from next boundary's `max(z)` or extended below.

## DWG Support

ezdxf reads DXF only. DWG requires the free ODA File Converter as an
external install. We detect `.dwg` extension and give a clear error with
instructions.

## Edge Cases

- Multiple polylines on same layer: merged and sorted by x
- Duplicate points: removed (within 1e-6 tolerance)
- Empty layers: warning, not error
- Missing mapped layer: warning for optional (GWT, nails), error for surface
- Zero-length nails: filtered out by SoilNail validation

## Dependencies

- `ezdxf >= 1.4` (optional, pip install ezdxf)
- Transitive: pyparsing, fontTools, typing_extensions
- numpy already in project deps

## References

- ezdxf documentation: https://ezdxf.readthedocs.io/
- DXF format reference: https://help.autodesk.com/view/OARX/2025/ENU/?guid=GUID-235B22E0-A567-4CF6-92D3-38A2306D73F3
- FHWA GEC-7 (Lazarte et al. 2003) — Soil Nail Walls (nail geometry)
