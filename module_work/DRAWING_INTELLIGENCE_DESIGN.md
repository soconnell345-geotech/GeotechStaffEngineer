# Drawing & submittal intelligence — design memo (2026-09-04)

Companion to DRAWING_INTELLIGENCE_TASK.md (the owner's spec). Research
basis: web+literature survey, code-verified against drawing_ir/
pdf_import/vision_tools (full findings in the 2026-09-04 research
report; key sources cited inline).

## The verdict the research delivers

**The owner's hypothesis is the published state of the art.** Hybrid
"deterministic geometry says WHERE, vision says WHAT" pipelines beat
pure-VLM approaches for engineering drawings; an independent 2026
benchmark showed the best frontier VLM reaching only ~80% on dimension
extraction with silent fabrication failures in several models, while
document-AI services (Azure Layout, Textract) "could not process"
drawing content at all. Academic SOTA for CAD symbol spotting treats
vector drawings as graphs of geometric primitives — the drawing_ir
architecture, independently validated. No public benchmark covers
civil/structural annotation constructs: we are building into a genuine
gap, validated by project fixtures rather than leaderboards.

## Architecture: two layers, one pattern

1. **Primitive layer** (exists = drawing_ir): Line/Polyline/Arc/Circle/
   Text with coords+layer+provenance+confidence from DXF (1.0),
   PDF-vector (1.0), raster (<1.0).
2. **Composition layer** (new): named annotation constructs assembled
   from primitives as confidence-scored PROPOSALS (the house
   `candidate_ground_surface` pattern — proposal_only, never asserted):
   leaders, dimensions, title blocks, revision clouds, keynote/detail/
   grid bubbles. DXF shortcut: LEADER/MULTILEADER/DIMENSION are
   first-class DXF entities (ezdxf exposes vertices, arrowheads, dogleg,
   linked annotation text, measurement values directly) — for DXF these
   skip composition entirely at confidence 1.0.

## The build list (no new external dependencies)

| # | Item | Owner module | Notes |
|---|---|---|---|
| B1 | DXF ingest: LEADER/MULTILEADER/DIMENSION (+ INSERT/ATTRIB for title-block attributes) as new IR entity kinds | drawing_ir | pure ezdxf API surfacing; from_dxf currently skips all four |
| B2 | `render_region(source, bbox, dpi, marks=None)` — the zoom-in vision primitive | vision_tools (+drawing_ir helper) | PyMuPDF `get_pixmap(clip=...)`; THE highest-leverage tool per CropVLM/agentic-zoom research |
| B3 | Set-of-marks overlay option on B2 (numbered marks at IR-known endpoints → converts "what is this pointing at" into multiple choice) | vision_tools | optional flag; A/B on fixture before default-on (SoM gains proven on natural images, plausible-not-proven transfer) |
| B4 | `entities_ending_near(ir, point, radius, type)` query + text-anchored search ("find text X → geometry terminating at it → other endpoint = points-at location") | drawing_ir.queries | closes the SAE loop; small addition |
| B5 | Composition functions: `find_leaders`, `find_dimensions`, `find_title_block_region`, `find_bubble_callouts`, `find_revision_clouds` | drawing_ir.queries | uniform across formats over shared primitives; rev clouds = lowest confidence tier (best-effort, drafting-practice-dependent) |
| B6 | Agent tool wiring: expose B2+B4+B5 as agent tools | funhouse_agent | the agent's loop becomes query→zoom→look→report |
| B7 | Raster-leg OCR decision: local OCR default (tenant DI High-Res OCR disabled), Azure DI as optional route for SCANNED title blocks/notes ONLY | drawing_ir.raster | DI verdict: OCR/tables only — categorically no geometry; never ask it to find constructs |

## Acceptance scenario: find-text-X (owner: "SAE" was an ARBITRARY example
## string — could be anything; nothing is built specific to it)

"Find all leaders whose tail text contains 'SAE'; report each callout's
location, the arrow-tip coordinates, and a vision-grounded description
of what it points at." Pipeline: text_items("SAE") →
entities_ending_near → leader proposal (or native DXF entity) →
render_region(tip bbox, marks) → vision describe → structured report.
Fixtures: one real DXF, one vector PDF sheet, one scanned sheet (owner
to supply representative sheets; PDF-vector leader recall is the
highest-uncertainty cell in the matrix — test it first). Additional
scenarios: extract this sheet's title block; list all revision-delta
callouts; enumerate keynote bubbles with their numbers and locations.

## Risks (from the research, kept honest)

- PDF-vector arrowhead recall unverified until a real fixture runs.
- Revision clouds genuinely uncertain in all formats (drafter-practice
  dependent) — scope as best-effort.
- Fixture-scoped accuracy claims only; no public benchmark exists.
- VLM fabrication risk on drawing content is DOCUMENTED in benchmarks —
  every composed construct stays a flagged proposal, and vision answers
  cite the zoomed region they were shown.

## FLEET REALITY (owner, 2026-09-04) — reprioritization

The team usually has NO DXF files — PDFs (vector or scanned) are the
working format, and producing DXFs manually is too arduous. Note also:
PDF->DXF conversion cannot help even in principle for annotations —
LEADER/DIMENSION entity semantics are destroyed at plot time (the PDF
contains only their geometry), so a converter returns plain lines, not
labeled entities. Direct PDF processing is therefore the PRIMARY path,
not a fallback. Priority order changes:
- PDF-VECTOR composition (B5, esp. find_leaders arrowhead heuristic) is
  now Phase 1 alongside B2 (region snip) + B4 (endpoint search).
- Raster/scan path (B7 + raster composition) rises — many team PDFs are
  scans.
- DXF-native ingest (B1) drops to opportunistic (still cheap; do it
  when convenient, mainly benefits any consultant-supplied DXFs).

## PRODUCT FRAMING (owner, 2026-09-04): chatbot-first, maximally flexible

End users are NOT tool-aware — they use the familiar chat interface:
"tell me X about this drawing", "find how many times Y occurs in this
drawing SET", "does this drawing set align with our standards
(standards provided separately)". Design consequences:
1. Every tool = a generic composable primitive the AGENT orchestrates;
   no per-question plumbing, all parameters runtime-supplied.
2. DRAWING SETS are first-class (Phase 2+): multi-page/multi-file
   ingest, per-sheet iteration, cross-sheet aggregation (counts,
   inventories, sheet-index awareness).
3. STANDARDS-CONFORMANCE scenarios (later phase): compare detected
   constructs/content against standards supplied as uploads or as
   reference-layer modules — the agent composes detection + reference
   lookup + judgment, with every claim citing the sheet region and the
   standard clause it compared against.

## Build plan

Phase 1 (B2+B4+B5-leaders): region-snip + endpoint search + PDF-vector
leader composition — the SAE scenario end-to-end ON A VECTOR PDF (the
fleet's real format). Phase 2 (rest of B5 + B6 + B7): full composition
family, agent wiring, raster/OCR leg. Phase 3 (B3 + B1): marks A/B +
opportunistic DXF-native ingest. Build on owner
word; needs 2-3 representative sheets from the owner's team as fixtures
(scrubbed of anything sensitive).

## Ground-truth fixture sources (scout survey, 2026-09-04)

Owner's insight: agencies publishing the SAME detail as CAD + PDF give
machine-readable ground truth (native LEADER/DIMENSION entities in CAD)
against the PDF as test input — real-drafting-practice scoring with no
manual labeling. Findings:
- **Key wrinkle: state DOTs are MicroStation DGN shops** (TxDOT, FDOT,
  Caltrans, WSDOT, NYSDOT, PennDOT, ODOT) — the free ODA File Converter
  does DWG<->DXF only, NOT DGN; DGN needs Bentley View export (free but
  manual) — a conversion tax.
- **Top picks (DWG-native, no login, direct ODA->DXF path):**
  1. Mecklenburg County NC Stormwater Services standard drawings
     (stormwaterservices.mecknc.gov/Standard-Drawings) — DWG+PDF pairs,
     drainage/culvert details with dimension callouts.
  2. Jacksonville FL City Standard Details (jacksonville.gov public
     works, "(dwg-pdf-formats)" page) — roadway/curb/drainage series.
  3. Caltrans 2025 Standard Plans — best structural/bridge leader
     density, but DGN (use only if the DWG sources prove insufficient).
- Pages bot-block scripted fetches (403 on WebFetch) — harvest the
  specific file links via a real browser session when Phase-1
  verification needs real-truth fixtures (5-10 pairs). DWG->DXF via ODA
  File Converter locally; then: parse DXF LEADER/DIMENSION entities =
  truth, run find_leaders on the paired PDF, score recall/precision.
- FloorPlanCAD (academic) = wrong format/domain; pattern reference only.
