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
  render.py    # render_region — the region-snip "zoom in" vision primitive
  tests/       # programmatic DXF/PDF/raster fixtures + query correctness
               # (leader_fixtures.py: synthetic multi-leader + decoy PDF sheets)
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
never applied*. PDF has no layers. Bezier curves are SAMPLED (8 subdivisions
per cubic, `pdf_import.extractor._sample_cubic_bezier`, Phase 2) — a drawn
circle arrives as a ~32-vertex circle-like ring and a cloud scallop keeps its
bump; before Phase 2 curves collapsed to their chord, which made curve-aware
construct detection impossible.

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
- `entities_ending_near(point, radius, entity_types=None)` — entities with an
  ENDPOINT (not just bbox/full-geometry) within radius; each hit carries
  `end` ("start"|"end"), `end_point`, and `other_end` (the far endpoint —
  e.g. where a leader points FROM). Circle/TextItem never match (no
  endpoints). The primitive "what terminates here" query.
- `text_anchored_geometry(pattern, radius=None)` — composes `text_items` with
  `entities_ending_near` around each match's insertion point: "find text X ->
  the geometry terminating there -> the far endpoint it points at"
  (`points_at`). `radius` defaults per-match from that text's own height.
  **PROPOSAL** (`proposal_only`) — adjacency is not proof of a leader
  relationship, confirm against the drawing.
- `candidate_ground_surface()` — **PROPOSAL only**: the widest left-to-right
  path (tie → upper). A heuristic suggestion the caller confirms, never an
  assertion. (Soil properties never come from a drawing.)
- `find_leaders(max_arrowhead_size=None, search_radius=None, text_radius=None,
  min_confidence=0.0)` — **PROPOSAL only**: composes a leader (bent shaft +
  arrowhead + tail text) from primitives — small closed 3-5-vertex
  Polyline/Region "arrowhead" candidates, the nearest Line/open-Polyline
  endpoint as the shaft, alignment of the shaft's terminal direction with the
  arrowhead's own apex-from-base direction, and the nearest TextItem to the
  shaft's far end as tail text. Confidence = weighted alignment + text
  proximity + shaft simplicity (see the docstring for the exact weights and
  the documented false-positive source: a dimension line's arrowheads are
  geometrically identical — a true dimension is a one-arrow leader
  geometrically and scores HIGH (~0.78), so the DOCUMENTED precision
  contract is `exclude_dimensions=True`, which lets `find_dimensions`
  claim those arrowheads first). Validated on synthetic PDF-vector
  fixtures (`drawing_ir/tests/leader_fixtures.py` + `test_find_leaders.py`):
  100% recall, 100% precision at confidence >= 0.5 *under
  exclude_dimensions* — fixture-scoped numbers, not a general benchmark
  claim.
- **Composition family (Phase 2)** — same proposal pattern
  (confidence + `evidence`, `proposal_only: True`, never asserted):
  - `find_dimensions()` — shaft with two DISTINCT arrowheads + roughly
    perpendicular extension lines + midpoint value text; straight shafts
    only (curved/angular dimensions out of scope). THE disambiguator for
    `find_leaders`' dimension false positive.
  - `find_title_block(edge_frac)` — edge-adjacent rectangle scored on
    edge adjacency + text density + rectangle nesting; returns the
    region bbox AND its text payload; text-cluster fallback (low
    confidence) when the sheet has no rectangles.
  - `find_bubble_callouts(max_radius, text_max_chars)` — native circles +
    circle-like closed rings (centroid circle-fit, rms/r <= 0.08) with
    short centered text; kinds keynote / grid_bubble (line ends on ring) /
    detail_callout (chord through center).
  - `find_revision_clouds(min_arcs)` — BEST-EFFORT tier (<= ~0.65 by
    design): native DXF Arc chains (endpoint union-find) + scalloped
    closed rings via the turn-angle cusp signature (smooth low-angle runs
    broken by opposite-sign junction spikes — empirically verified against
    PyMuPDF scallops; a rounded rectangle's same-sign corners are
    rejected); plus revision DELTAS (small labelled triangle with NO
    shaft terminating at it).
- **Performance (real-sheet scale)**: the composition loops index entity
  endpoints in a uniform grid (`_EndpointGrid`) and use O(1) id lookup —
  a 10k-entity Mecklenburg sheet runs in seconds (a linear-scan
  implementation profiled at minutes). Arrowhead candidates are gated on
  non-degeneracy (area/perimeter² >= 0.02) and shafts on
  length-vs-arrowhead scale, which keeps SHX glyph strokes from flooding
  the proposals; `_default_max_arrowhead_size` carries a page-diagonal
  floor because glyph-stroked sheets collapse the median segment length.
- `summary_stats()` — counts by type/layer, page metadata, extent, scale;
  reports `has_text` + an explicit note when a sheet has NO extractable
  text (SHX/stroked lettering — verified on all 10 Mecklenburg plots):
  text queries return nothing there and a zero count is inconclusive.

### The region-snip vision primitive (`render.py`)

`render_region(filepath|content, page, bbox=None, dpi=300, pad_frac=0.15,
marks=None)` renders a zoomed-in PNG crop of a PDF page — turns a WHERE
(a bbox from any exact source, typically a query result) into a high-DPI
image a vision model can answer WHAT about, instead of feeding a whole page
(illegible small annotations) or asking the model to guess pixel coordinates.
`marks` draws numbered circles at given points for set-of-marks prompting.

**Coordinate contract**: `bbox`/`marks` are PDF points in PyMuPDF's own page
space — origin top-left, y DOWN (`page.rect`/`fitz.Rect` convention) — NOT
the `origin="bottom_left"` convention `from_pdf_vector` defaults to. Convert
an IR point before calling: `x_pdf = x_ir; y_pdf = page_height_pt - y_ir`.

Implemented in `funhouse_agent/vision_tools.py` as `_dispatch_render_region`
(same conventions as `_dispatch_analyze_pdf_page`) plus a save-to-file
counterpart `render_region_to_file`. **Live on every agent surface since
Phase 2 (B6)**: `EXTENDED_TOOLS` + `VISION_TOOL_DESCRIPTIONS` + the
dispatch route (v1 agent), `make_vision_tools` (deep agent), and
`OPENAI_TOOLS`/`EXTENDED_TOOL_NAMES` (native). The drawing_ir adapter's
`snip_region` method is the save-to-disk twin and converts IR bottom-left
coordinates to this frame for the caller.

### A verified ingest fact `find_leaders` depends on

`from_pdf_vector` **never emits a `Region`** entity (only Line/Polyline/
TextItem — see `ingest.py`'s `from_pdf_vector` loop). A filled-triangle
arrowhead therefore arrives as a **closed Polyline with 3 vertices and a
small bbox** — verified empirically by round-tripping a synthetic PyMuPDF
leader through `from_pdf_vector` (see `drawing_ir/tests/leader_fixtures.py`).
Gotcha along the way: PyMuPDF's `Shape.finish()` defaults `closePath=True`
even for plain multi-segment line-work, so a NAIVELY-drawn shaft (without
`closePath=False`) also comes back `closed=True` — geometrically
indistinguishable from an arrowhead by the `closed` flag alone. Real CAD/plot
PDF exporters normally leave a stroke-only path open, and the fixture
generator draws shafts with `closePath=False` explicitly to match; the
`closed` flag alone is therefore NOT a reliable arrowhead signal on its
own — `find_leaders` also gates on vertex count (3-5) and bbox size.

## Funhouse adapter

`funhouse_agent/adapters/drawing_ir_adapter.py` exposes five methods and
**caches the IR server-side keyed by a `handle`** (a full IR can be large):

- `digitize_drawing(file_path, source=auto|dxf|pdf_vector|raster, …)` → `handle`
  + summary/stats (+ PDF `scale_candidates`, `has_text`). The full IR is
  never returned.
- `query_drawing(handle, query, params)` → one slice (`allowed_values` on the
  query name; per-query required/allowed params validated). Includes the
  whole composition family + `entities_ending_near`/`text_anchored_geometry`.
- `get_entities(handle, ids)` → exact coordinates for specific ids.
- `snip_region(file_path, output_path, bbox, frame=ir|pdf, marks, …)` →
  zoomed PNG crop on disk (converts IR bottom-left coords to the render
  frame; set-of-marks supported) for a follow-up `analyze_image`, or use
  the one-step `render_region` vision tool.
- `search_drawing_set(file_paths, pattern|construct, pages, min_confidence)`
  → every page of one or more PDF/DXF files: per-file/per-page counts +
  compact match locations ("how many times does X occur in this set");
  pages with no text layer are flagged `no_text_layer` (zero counts there
  are inconclusive — SHX).

`drawing_ir` is registered in `MODULE_REGISTRY`, so it is a directly-callable
analysis-layer tool for the primary agent (it is an I/O tool, not a reference).

## Downstream

The IR is a superset of what `geo_project` / `slope_stability` / `fem2d`
ingestion needs. Wiring `geo_project` ingestion to consume a confirmed IR (with
its provenance quarantine for anything below confidence 1.0) is the natural
follow-up — the schema already carries the provenance + confidence that
quarantine keys on.
