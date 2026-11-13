# DXF Import Module — Design Notes

## Purpose

Import CAD cross-sections (DXF format from AutoCAD / Civil 3D) and translate
them into `SlopeGeometry` for slope stability analysis. DXF provides geometry
only — soil properties must come from the user.

## Three-Step Workflow

1. **`discover_layers(filepath)`** — Read DXF, list all layers with entity
   counts, types, sample text, and bounding boxes.
2. **`parse_dxf_geometry(filepath, layer_mapping, units)`** — Extract
   coordinates from user-mapped layers, convert units, produce sorted profiles.
3. **`build_slope_geometry(parse_result, soil_properties)`** — Assemble
   `SlopeGeometry` with user-supplied soil parameters.

## DXF Entity Mapping

| Entity Type | Used For | Notes |
|-------------|----------|-------|
| LWPOLYLINE | Surface, boundaries, GWT | Primary geometry. Vertices extracted. |
| POLYLINE | Surface, boundaries, GWT | 3D polylines also supported. |
| LINE | Surface (merged), boundaries, **nails** | Individual segments merged for profiles. Each nail = one LINE from head to tip. |
| SPLINE | Surface, boundaries | Flattened to polyline via `ezdxf.flattening()`. |
| TEXT / MTEXT | Annotations | Insertion point + text content extracted. |

## Layer Name Convention

DXF layer names are arbitrary — every CAD operator names them differently.
That's why we use a discover-then-parse workflow: the user inspects layer
names and creates a `LayerMapping`.

## Unit Handling

- DXF `$INSUNITS` header variable provides a hint (detected automatically)
- User specifies `units=` parameter to confirm
- All output in SI meters
- Supported: m, mm, cm, ft, in

## Nail Convention

- Each LINE on the nail layer = one nail
- Head = leftmost endpoint (closer to slope face)
- Tip = rightmost endpoint (deeper into slope)
- Inclination computed from head/tip geometry (degrees below horizontal)
- User supplies nail defaults (bar diameter, bond stress, etc.) via `nail_defaults` dict

## Layer Stacking

For multi-layer slopes (`build_slope_geometry`), each boundary is the TOP
of the soil it names: that soil lies below the boundary, down to the next
boundary. The soil between the ground surface and the highest boundary
takes the soil property named "Surface", or else the first property that
no boundary names.
- Surface profile = top of the top soil ("Surface", or the first property
  no boundary names)
- Boundary 1 (the highest) = bottom of the top soil / top of the soil
  boundary 1 names
- Boundary 2 = bottom of that soil / top of the soil boundary 2 names
- etc.

Layer elevations: `top_elevation = max(z)` along top profile,
`bottom_elevation` from next boundary's `max(z)` or extended below.

One material per point (slope_stability, GM-140 step 1, 2026-11-13): in
`build_slope_geometry` the layer named for a boundary has that boundary as
its `top_boundary_points` and the next boundary down as its
`bottom_boundary_points` (the lowest layer ends 2 m below its boundary's
lowest point), so no ground is held by two layers and none is left without
one. Before, each such layer's bottom was its own boundary: the ground
under a boundary had no layer and the model ended at the lowest boundary.

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
