# Drawing & submittal intelligence — task spec (owner, 2026-09-03)

**Status: QUEUED — next major task after the structural ref campaign
waves in flight. Research first, then design/build.**

## The owner's problem statement (near-verbatim)

Structural engineers on the team want the agent to handle construction
drawings and submittals — documents needing BOTH text and visual
processing. Motivating example: find ALL instances where a **leader**
(AutoCAD term: an arrow with a bend, text at the tail) points to
something with the abbreviation **"SAE"** at the tail; report (a) exact
callout locations, (b) some attempt at describing what each leader
points AT. Feeding a whole-page image to the model worked poorly —
consistent with research that LLMs are good at image patterns but bad
at raw geometry.

## SCOPE CLARIFICATION (owner, 2026-09-04): GENERAL, not leader-specific

The SAE-leader hunt is an EXAMPLE, not the scope. The target is generic
drawing-annotation understanding composed from primitives. Owner's
enumerated examples of visually-detectable constructs:
- **leaders/arrows** (the original example)
- **title blocks** (sheet metadata: number, title, revision, scale, firm)
- **dimensions** (dimension lines, extension lines, dimension text)
- **revision bubbles and markers** (clouds, deltas/triangles w/ numbers)
- **general visual keys** — e.g., a number in a circle (keynotes, detail
  callouts, section marks, grid bubbles)
Owner's architectural point (verbatim spirit): "if it can pick up lines,
arcs, circles, arrows, shapes and the location and orientation of text,
it can piece a lot of this together" — i.e., a PRIMITIVE-EXTRACTION layer
plus a COMPOSITION layer that assembles primitives into named annotation
constructs. The research and design must treat annotation constructs as
a pluggable family over shared primitives, not one-off detectors.

## Owner's initial ideas (evaluate all)
- Azure **Document Intelligence** as an agent tool when asked to review
  a drawing (note: prior finding — High-Res OCR add-on DISABLED in the
  Databricks tenant; folder-14 bake-off was already a FUTURE_IDEAS item;
  TinyApps/Key-Vault route may differ).
- A tool SUITE rather than one tool:
  - snip a region of a PDF and process it as an image (zoom-in vision)
  - pull structured data on lines/circles/arrows (vector geometry)
  - "idk anything else" — survey what else belongs (his words: worth
    seeing if there's research out there on how to handle it).

## What we ALREADY have (start here, don't rebuild)
- **drawing_ir/** (61 tests): "deterministic extractor owns coordinates,
  LLM owns semantics." Unified IR (Line/Polyline/Arc/Circle/Text/Region
  w/ coords+layer+provenance+confidence) from DXF (ezdxf, conf 1.0),
  PDF-vector (pdf_import, conf 1.0), raster (OpenCV Hough/contour,
  conf<1). Slice-query interface (bbox/angle/text/layer/nearest) — the
  LLM requests slices, not pixels. THE architectural foundation.
- pdf_import/ (vector extraction, scale calibration, vision grid
  overlay, vision<->vector cross-check), dxf_import/, vision tools
  (analyze_image, analyze_pdf_page, read_reference_figure renders pages
  at 220 dpi).
- Gaps vs the SAE use case (first-pass guesses, verify in research):
  (1) no LEADER/MULTILEADER entity support in the IR (ezdxf exposes
  these directly for DXF; PDF-vector leaders = polyline + arrowhead
  triangle clusters — detection needed); (2) no "render THIS bbox at
  high dpi and vision-analyze it" agent tool (region-snip = the
  killer primitive: tip coordinates from geometry -> zoomed image ->
  'what is this pointing at'); (3) no text-anchored geometry search
  ("find text 'SAE', give me the connected leader, give me the tip
  coords"); (4) submittals are usually SCANNED (raster leg + OCR
  quality matters — Document Intelligence bake-off relevant).

## Research questions for the survey (web + literature)
1. State of the art on LLM/VLM engineering-drawing understanding
   (P&ID/blueprint/floor-plan parsing literature; "symbol spotting";
   arrow/leader detection papers; vision-grounding with coordinate
   overlays / set-of-marks prompting; agentic zoom-in loops).
2. Azure Document Intelligence Layout/Read vs drawings specifically —
   does it return line/arrow geometry or only text+tables? (Suspect:
   text/tables/selection marks only — verify.) Competitors worth a
   look for completeness (Google DocAI, AWS Textract) though Azure is
   the sanctioned one.
3. Established open tooling: ezdxf LEADER/MLEADER API surface, PyMuPDF
   drawing-command extraction for arrowhead clustering, OpenCV arrow
   detection approaches for raster.
4. Coordinate-communication patterns that work for LLMs: grid overlays
   (we have one), tile-by-tile sweep protocols, hybrid "geometry says
   WHERE, vision says WHAT" loops — what does the research support.

## Deliverable shape
A design memo (module_work/) proposing the tool suite (agent-facing
methods + which existing module owns each), the SAE-callout workflow as
the acceptance scenario end-to-end, and a build plan with validation
strategy (a real DXF + a vector PDF + a scanned sheet as fixtures).
Then build on owner word.
