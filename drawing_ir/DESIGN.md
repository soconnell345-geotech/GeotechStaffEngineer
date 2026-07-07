# Drawing IR — design notes

## North star

> "LLM vision often doesn't handle precise geometry well."

So the division of labor is deliberate: the **deterministic extractor owns
coordinates**; the **LLM owns semantics**. A drawing is digitized once into a
single, exact intermediate representation (the `DrawingIR`), and the model then
asks *structured questions* of it — "what text sits near entity `e12`?", "which
polylines run left-to-right longer than 20 m?" — instead of eyeballing pixels
and guessing numbers. Every coordinate the LLM eventually uses came from the
extractor, not from a vision read.

This module is the schema + the ingest legs + the query surface that make that
possible. It does **not** interpret the drawing (no "this is a retaining wall").
It hands an LLM/agent a clean, queryable geometry model and lets *that* layer do
the interpretation, with the raw coordinates always one `get_entities` call away.

## What it is

```
drawing_ir/
  results.py   # DrawingIR + Entity types (Line/Polyline/Arc/Circle/TextItem/Region)
  ingest.py    # from_dxf / from_pdf_vector / from_raster
  raster.py    # the OpenCV tracing leg (isolated so cv2 stays optional)
  queries.py   # the LLM-facing slice queries
  tests/       # programmatic DXF/PDF/raster fixtures + query correctness
```

## The IR schema (`results.py`)

A `DrawingIR` is **one drawing page**: page metadata + a flat list of entities.
It round-trips to/from JSON losslessly (`to_dict` / `from_dict`,
`entity_from_dict`).

Page metadata: `width`, `height`, `units`, `coordinate_space`, `scale`,
`scale_provenance`, `origin`, `source`, `warnings`, `metadata`.

Every entity carries a common envelope:

| field        | meaning |
|--------------|---------|
| `id`         | stable within the page (`e0`, `e1`, …) |
| `layer`      | CAD layer / logical group (DXF only), else `None` |
| `color`      | hex `#rrggbb`, or `ACI<n>` when only a DXF color index is known |
| `style`      | linetype / a note like `approx_from_spline`, `hough`, `contour` |
| `source`     | `dxf` \| `pdf_vector` \| `raster_trace` (provenance) |
| `confidence` | `1.0` for deterministic sources; `< 1.0` for raster detections |
| `bbox`       | `(x_min, y_min, x_max, y_max)`, auto-computed from geometry |

Concrete types and their geometry:

- **Line** — `start`, `end` (+ derived `length`, `angle_deg` folded to [0,180)).
- **Polyline** — `vertices`, `closed` (+ `length`).
- **Arc** — `center`, `radius`, `start_angle`, `end_angle` (CCW degrees). The
  bbox is sampled (it accounts for the axis crossings the arc actually sweeps).
- **Circle** — `center`, `radius`.
- **TextItem** — `content`, `position` (insertion point), `rotation`, `height`.
  Its bbox is an **approximation** (width estimated from the character count) —
  true text metrics are font-dependent and not recovered.
- **Region** — `boundary` ring + optional `pattern` (hatch/filled area).

### Coordinate space & the "flag"

All entities on a page share ONE coordinate space, flagged on the page (not
repeated per entity):

- `coordinate_space="model"` → calibrated engineering units (`units`, SI meters
  by house convention); a scale has been applied. `scale` (model units per page
  unit) and `scale_provenance` say how.
- `coordinate_space="page"` → raw page/pixel units (`units` = `pt` for PDF
  points, `px` for raster pixels, or the DXF drawing units); no calibrated scale.

`origin` records the y convention (`bottom_left` = engineering up-positive, the
default for PDF/raster; DXF model space is already up-positive).

## Provenance & confidence semantics

Confidence is **the honesty knob**. Deterministic sources read exact path
coordinates and get `1.0`. The raster leg only sees pixels, so every entity is a
*detection* with a fixed sub-1.0 tier (see below). An agent should treat a
`confidence < 1.0` entity as a proposal to confirm, and always prefer
`get_entities` coordinates over its own pixel reading.

| source          | confidence | notes |
|-----------------|-----------|-------|
| `dxf`           | 1.0 | exact CAD geometry, with layers/colors |
| `pdf_vector`    | 1.0 | exact PDF path coordinates |
| `raster_trace`  | 0.4–0.6 | Hough lines 0.6, circles 0.5, contours 0.5, OCR 0.4 |

## Ingest legs (`ingest.py`)

### `from_dxf` (ezdxf) — confidence 1.0
Direct model-space pass: LINE, LWPOLYLINE/POLYLINE, ARC, CIRCLE,
ELLIPSE/SPLINE (flattened to polylines, tagged `approx_from_*`), TEXT/MTEXT, and
best-effort HATCH → Region. Coordinates are converted to **SI meters** using the
drawing's `$INSUNITS` header (or a supplied `units`); `scale`/`scale_provenance`
record any unit conversion. Layers and colors are preserved.

### `from_pdf_vector` (PyMuPDF) — confidence 1.0
Reuses `pdf_import.extract_colored_paths` (per-path point lists + color) and
`pdf_import.discover_pdf_content` (page size + text). Each path becomes a Line
(2 points) or Polyline; each text span a TextItem. With an explicit `scale`
(m per point) or a two-point `calibration` (`{p1, p2, distance_m}` via
`pdf_import.calibrate_scale`), coordinates are promoted to model meters;
otherwise the IR stays in page points and **scale candidates** parsed from the
page text (`pdf_import.propose_scale`) are attached to `metadata` as *proposals,
never applied*. PDF has no layers; bezier curves are represented by their
chord polyline (an honest limitation).

### `from_raster` (OpenCV) — confidence < 1.0
Delegates to `drawing_ir.raster.trace_raster` (keeps `cv2` optional). See below.

### The cross-import ruling

`ingest.py` imports the `dxf_import` / `pdf_import` I/O modules directly. This is
consistent with the house convention, **not** a violation of it: the
"no cross-module imports" rule targets the 30 computational *analysis* modules
(so they stay independently testable). The **I/O modules already form a
dependency layer** — `pdf_import` imports `dxf_import.converter`, and
`geo_project/ingest.py` imports both `dxf_import` and `pdf_import`. `drawing_ir`
joins that same I/O layer with the same pattern. `results.py` and `queries.py`
are kept pure-schema (no module imports) so the schema/query core has zero
heavy dependencies.

## The raster leg's honest limits (`raster.py`)

The raster leg is the low-confidence bootstrap, not a replacement for vector
data. What it does and does **not** do:

- **Lines** — probabilistic Hough on Canny edges. A *thick* stroke has two
  edges, so one drawn line can trace as two nearly-parallel Hough segments.
  Coordinates are pixel-quantized.
- **Circles** — Hough gradient transform. Sensitive to `param2`/radius bounds.
- **Contours** — external-contour tracing + polygon approximation → closed
  Polylines. Best for filled/outlined *regions*; on pure line-work it may trace
  the *outline* of a stroke, overlapping the Hough line result — so the
  detectors are individually toggleable (`detect_lines/circles/contours`).
- **Arcs are not recovered** — a partial curve is missed or seen as a contour.
- **No layers** — raster has none; `layer` is always `None`. Colors are sampled
  per detection from the source pixels.
- **Text via OCR is opt-in and best-effort** — only if `pytesseract` + a
  Tesseract binary are importable/working. If not, text is **skipped with a
  warning** (positions are never invented). This machine has no Tesseract, so
  the raster text leg degrades to positions-only there.

Because tracing is inexact, the raster tests are tolerance-based (counts and
approximate coordinates, not exact equality).

## Query interface (`queries.py`) — the LLM's surface

Every function takes a `DrawingIR` and returns compact, JSON-able results —
entity **references** (`id` + a small summary), never full coordinate dumps.
The agent narrows with queries, then pulls exact coordinates for a shortlist via
`get_entities`.

- `entities_in_bbox(x_min,y_min,x_max,y_max, mode, entity_type)` — spatial window
- `nearest_entity(x,y, entity_type, k)` — k nearest by true point-to-geometry
  distance (point-to-segment / point-to-ring / insertion point)
- `lines_by_angle(min_deg,max_deg)`, `horizontal_lines(tol)`,
  `vertical_lines(tol)` — over Line **and** Polyline segments
- `polylines_longer_than(min_length)`
- `text_items(pattern)` (regex or literal substring), `text_near(entity_id,
  radius)`
- `entities_on_layer(layer)`, `entities_by_color(color)`
- `get_entities(ids)` — full exact coordinates for a shortlist
- `candidate_ground_surface()` — **PROPOSAL only**: the widest left-to-right
  path (tie → upper). A heuristic suggestion the caller confirms, never an
  assertion. (Soil properties never come from a drawing.)
- `summary_stats()` — counts by type/layer, page metadata, extent, scale

## Funhouse adapter

`funhouse_agent/adapters/drawing_ir_adapter.py` exposes three methods and
**caches the IR server-side keyed by a `handle`** (a full IR can be large):

- `digitize_drawing(file_path, source=auto|dxf|pdf_vector|raster, …)` → `handle`
  + summary/stats (+ PDF `scale_candidates`). The full IR is never returned.
- `query_drawing(handle, query, params)` → one slice (`allowed_values` on the
  query name; per-query required/allowed params validated).
- `get_entities(handle, ids)` → exact coordinates for specific ids.

`drawing_ir` is registered in `MODULE_REGISTRY`, so it is a directly-callable
analysis-layer tool for the primary agent (it is an I/O tool, not a reference).

## Downstream

The IR is a superset of what `geo_project` / `slope_stability` / `fem2d`
ingestion needs. Wiring `geo_project` ingestion to consume a confirmed IR (with
its provenance quarantine for anything below confidence 1.0) is the natural
follow-up — the schema already carries the provenance + confidence that
quarantine keys on.
