# dxf_export — Design Notes

## Purpose

Export cross-section geometry to DXF files. Complements `dxf_import` and
`pdf_import` to enable round-trip workflows: PDF/DXF → extract → analyze →
export new DXF.

## Layer Conventions

| Layer            | Color | Entity     | Source field          |
|------------------|-------|------------|-----------------------|
| SURFACE          | 3     | LWPOLYLINE | surface_points        |
| BOUNDARY_<name>  | 1     | LWPOLYLINE | boundary_profiles     |
| GWT              | 4     | LWPOLYLINE | gwt_points            |
| NAILS            | 6     | LINE       | nail_lines            |
| ANNOTATIONS      | 7     | TEXT       | text_annotations      |

Colors follow AutoCAD ACI standard: 1=red, 3=green, 4=cyan, 6=magenta, 7=white.

## Coordinate System

- Coordinates are (x, z) matching the geotechnical convention used by
  `dxf_import` and `slope_stability`.
- DXF writes to the XY plane — the z coordinate maps to DXF's Y axis.

## Units

The `$INSUNITS` header is set based on the `units` parameter:
- 6 = meters (default)
- 2 = feet
- 4 = millimeters
- 5 = centimeters
- 1 = inches

## DXF Version

Default is `R2010` — widely compatible with AutoCAD 2010+ and most DXF readers.

## Input Formats

### Raw geometry (export_to_dxf / to_dxf_bytes)
- `surface_points`: `[(x, z), ...]`
- `boundary_profiles`: `{"Clay": [(x, z), ...], ...}`
- `gwt_points`: `[(x, z), ...]` or None
- `nail_lines`: `[{"x_head": ..., "z_head": ..., "x_tip": ..., "z_tip": ...}, ...]`
- `text_annotations`: `[{"text": "...", "x": ..., "y": ...}, ...]`

### Parse result (export_parse_result)
- Accepts `DxfParseResult` or `PdfParseResult` directly
- PdfParseResult is auto-converted via `pdf_import.to_dxf_parse_result()`

## Dependencies

- `ezdxf >= 1.4` (same optional dependency as `dxf_import`)
