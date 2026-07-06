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

## Scale calibration (C1, v5.3) — `scale.py`
Two ways to establish the `extract_vector_geometry(scale=...)` factor
(meters per drawing unit / PDF point):
- `calibrate_scale(p1, p2, distance_m)` — DETERMINISTIC two-point calibration
  from two drawing points and their known real-world separation. Exact, no
  plot-size assumption; the reliable path.
- `parse_scale_annotations(text_blocks)` / `propose_scale(...)` — parse scale
  ANNOTATIONS ("SCALE 1:100", '1" = 20 ft', "1 cm = 2 m") into scale CANDIDATES,
  each tagged with `basis`, `provenance`, `confidence`, and `applied: False`.
  **Proposals only — never silently applied** (the ratio/engineering candidates
  assume the PDF is at true plot size, 1 pt = 1/72 in, which should be
  spot-checked against a known dimension). The caller/user confirms one and
  passes its `scale_factor` to the extractor. Funhouse adapter: `calibrate_scale`,
  `propose_scale`.

## Label -> region association (C2, v5.3) — `labels.py`
Proposes a colour->role `role_mapping` from the drawing's text labels:
- `classify_label(text)` — text -> role ("surface"/"gwt"/"boundary_<Name>").
  GWT/surface phrases first; then a USCS 2-letter symbol (SM/CL/…); else the
  RIGHTMOST soil noun wins ("silty SAND" -> Sand, "sandy CLAY" -> Clay).
- `associate_labels_to_regions(regions, text_blocks)` — attach each label to a
  region. Soil (area) labels use the ENCLOSING region (point-in-polygon); GWT and
  surface (LINE) labels use the nearest polyline edge (so a water-table label
  sitting inside a soil box still binds to the nearby line, not the box).
- `propose_role_mapping(regions, text_blocks)` — build `{hex_color: role}` from
  the closest label per colour. **Proposal only** (`applied: False`); confirm
  before passing to `extract_vector_geometry(role_mapping=...)`. `regions` come
  from `extract_colored_paths(...)` (one entry per vector path: `{color, points}`).
  Funhouse adapter: `propose_role_mapping` (chains PDF -> paths+text -> mapping).

## Geometry cleanup (C3, v5.3) — `cleanup.py`
Tolerance-parametrized, pure-geometry cleanup to run BEFORE
`build_slope_geometry` / `build_fem_inputs`:
- `dedupe_consecutive(points, tol)` — drop consecutive near-duplicate vertices.
- `merge_collinear(points, angle_tol_deg)` — thin a densely-sampled straight run
  to its endpoints while KEEPING corners (a vertex is kept only when the path
  bends by more than `angle_tol_deg`).
- `snap_endpoints(polylines, tol)` — snap near-coincident polyline ENDPOINTS to a
  shared averaged vertex (interior vertices untouched) so touching lines meet.
- `join_polylines(polylines, tol)` — greedily join segments sharing endpoints
  (with reversal) into longer chains.
- `cleanup_polyline(...)` / `cleanup_geometry(surface, boundaries, gwt, ...)` —
  apply dedupe + collinear-thin per polyline, cross-snap surface+boundary
  endpoints, optionally join; returns cleaned copies (inputs untouched) + a
  before/after point-count report. Funhouse adapter: `cleanup_geometry`.

## Vision grid overlay (C4, v5.3) — `vision.py`
`extract_geometry_vision(..., grid_overlay=True, grid_spacing=...)` renders the
PDF page with a labelled coordinate grid (`render_page_with_grid`) before the
vision call and uses the grid-aware prompt (`GRID_VISION_PROMPT`), so the model
reads coordinates OFF the grid instead of guessing — improving read-off accuracy.
The grid is labelled in drawing units (PDF points; x along the bottom, z from the
bottom-left upward), matching the `extract_vector_geometry` frame so vision and
vector read-offs are directly comparable (feeds C5 cross-check). Default
(`grid_overlay=False`) is unchanged. The grid is drawn on the in-memory page and
never saved. (Not a flat-JSON adapter method — the vision path needs an `image_fn`
engine.)

## Vision <-> vector cross-check (C5, v5.3) — `crosscheck.py`
`cross_check(vector_result, vision_result, tol)` compares the two extractions
feature by feature (surface, each boundary, gwt) and returns a DISCREPANCY
REPORT: which features are present in both / only one, and the vertical
deviation (`polyline_deviation` -> RMS / max / mean over the overlapping x-range)
between the polylines. `agree` is True only when every shared feature is within
`tol` and no feature is present in only one extraction. Nothing is auto-merged —
the report is for the caller to reconcile (geo_project provenance-quarantine
spirit). Funhouse adapter: `cross_check`.

## References

- PyMuPDF documentation: https://pymupdf.readthedocs.io/
- PDF specification ISO 32000
