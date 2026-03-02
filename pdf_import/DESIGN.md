# PDF Import Module — Design Notes

## Purpose

Extract cross-section geometry from PDF drawings (Plaxis, Slope/W, hand sketches)
for use in slope_stability and fem2d analyses. Two extraction methods:

1. **Vector extraction** — PyMuPDF `page.get_drawings()` for PDF vector paths
2. **Vision extraction** — LLM image analysis via pluggable `image_fn`

## Data Pipeline

```
PDF file ──→ discover_pdf_content()     → content inventory
         ──→ extract_vector_geometry()  → PdfParseResult
         ──→ extract_geometry_vision()  → PdfParseResult

PdfParseResult ──→ to_dxf_parse_result() ──→ build_slope_geometry()
                                          ──→ build_fem_inputs()
```

## Coordinate System

PDF coordinates: origin at bottom-left, Y up (same as engineering convention).
All output in meters (user provides scale factor if drawing units differ).

## Role Mapping (Vector Extraction)

Maps PDF drawing attributes (stroke color, line style) to geometric roles:
- `"surface"` — ground surface polyline
- `"gwt"` — groundwater table
- `"boundary_<name>"` — soil boundary (e.g., `"boundary_Clay"`)

## Dependencies

- **PyMuPDF** (`fitz`) — optional, same pattern as ezdxf in dxf_import
- No dependency on any specific LLM provider — `image_fn` is a callable

## Units

All output coordinates in meters (SI). Scale factor converts from drawing units.

## References

- PyMuPDF documentation: https://pymupdf.readthedocs.io/
- PDF specification ISO 32000
