# Drawing & submittal intelligence — design memo (2026-09-04)

> **PHASE 3: COMPLETE + INDEPENDENTLY VERIFIED (ship-with-fixes,
> fixes applied; planlens commits 5b7147c/3aae54f + close-out,
> 2026-09-04/05).** What the verifier's measurements say actually
> moved the numbers (attribution matters — pitch honestly):
> (1) **The transform-fit discovery is the win.**
> `planlens.ir.align.fit_plot_transform` (rotation/scale/offset
> anchor-voting) revealed the ground-truth plots are ROTATED 270 at
> exactly 72 pt/in (rms 0.02-0.03 pt, 100% anchors matched). Under
> the corrected transform plus fold-blind triangle alignment, leader
> recall is **11/25** — and the verifier proved triangles-alone score
> the SAME 11/25: the new fill-cluster arrowhead model contributes
> ZERO marginal recall today. It ships as tagged groundwork
> (`arrowhead_kind: fill_cluster`, density/aspect gates) for the
> sparse-dot regime; residuals are diagnosed per tip in
> score_compositions.py (4 tips have NO plotted arrow fragments; the
> rest sit in stipple below any principled density gate — verifier
> measured the distributions and concurred). Dimensions stay 1/16,
> and on stipple no-text sheets find_dimensions is measured ~ZERO
> precision at default confidence (44 proposals vs 10 truth on 3001,
> none near truth) — flagged in the adapter WARNING; real-sheet
> dimension output = noise until the split-shaft v2 (native dims plot
> as TWO collinear one-arrow halves around a text gap) + a no-text
> confidence cap land.
> (2) **B7 OCR leg is solid and fairly stated** — `planlens.ocr`
> (extra `planlens[ocr]`, RapidOCR/onnxruntime, permissive licenses,
> models in-wheel, no runtime downloads; ~170 MB clean-env incl. the
> full-opencv dependency — see README for the opencv-variant caveat):
> render → OCR → TextItems mapped into the IR frame. Coverage 88-92%
> per sheet held up under independent rematching (exact-only is
> 22-45% — "partial" containment matches carry the headline; medians
> 1.4-7 pt are convention-dependent). Committed check:
> ocr_coverage_check.py. Verifier property-tested _unrotate_px exact
> for all four rotations; auto-rotation now probes all four (a
> cls-flip residual for tied 180 pairs is documented in ocr.py).
> Agent wiring: `digitize_drawing(ocr_text=true)` +
> `search_drawing_set(ocr_text=true)` live-verified on the no-text
> 21.01 sheet; set-level IR cache now lock-guarded (verifier caught a
> concurrent double-augment race). **Deferred: B1 DXF-native ingest;
> find_dimensions split-shaft v2 + no-text confidence cap; B3 marks
> A/B; fit_plot_transform degenerate-scale extent guard (docstring
> caveat in place); opencv full-vs-headless resolution.**

> **PLANLENS SPLIT (2026-09-04, owner-named):** the code this memo
> describes now lives in the separate `planlens` package repo
> (../planlens; import `planlens.ir` / `planlens.pdf` /
> `planlens.dxf`). Module paths below (drawing_ir/, pdf_import/) are
> historical. Phase 3 builds in planlens; this memo remains the plan
> of record. No OBO branding in the package (owner).

> **PHASE 2: COMPLETE (commits 8a817fa + b050124 + the hardening commit,
> 2026-09-04).** Shipped: (1) bezier-sampled PDF ingest (circles/scallops
> survive as curves); (2) the full composition family — find_dimensions
> (+ the leader<->dimension disambiguation via
> `find_leaders(exclude_dimensions=True)`, now the DOCUMENTED precision
> contract: a dimension is geometrically a one-arrow leader and scores
> ~0.78 unfiltered), find_title_block, find_bubble_callouts,
> find_revision_clouds (best-effort tier) — all confidence+evidence
> proposals; (3) B6 agent wiring on every surface: render_region vision
> tool (v1/deep/native), 7 new query_drawing queries, snip_region
> (IR->PDF frame conversion + marks), search_drawing_set (multi-page/
> multi-file counts — the "how many times does X occur in this set"
> primitive); (4) SHX no-text-layer reporting end-to-end (has_text /
> no_text_layer flags — zero text counts on SHX sheets are called
> inconclusive); (5) real-sheet performance (endpoint grid + id map:
> 10k-entity sheets in seconds, was minutes) and glyph-flood hardening
> (non-degenerate arrowheads, shaft straightness/scale gates: 5/10 real
> sheets report zero spurious dimensions). **Real-truth baseline run**
> (module_work/drawing_ground_truth/score_compositions.py): leader/dim
> tip recall vs native truth 0/25, 0/16 — Mecklenburg plots render
> arrowheads as MICRO-DOT FILL CLUSTERS (~0.06-pt segments), a
> representation the triangle model cannot see; that + SHX text = the
> two Phase-3 legs (fill-cluster arrowheads, B7 raster/OCR). Bubbles:
> 40/40 count match on 10.31A. Numbers are the baseline — grow the
> representation model, don't tune to them.

> **PHASE 1: COMPLETE (committed d6bccf4, 2026-09-04).** render_region +
> entities_ending_near/text_anchored_geometry + find_leaders shipped with
> synthetic-fixture validation (100% recall, 100% precision@0.5;
> dimension-arrowhead decoys pinned ~0.3 as the documented false-positive
> source). Fable-verified; a model-space radius unit bug was caught and
> fixed pre-commit. Phase 2 = agent wiring + remaining composition family
> + drawing sets + real DWG+PDF ground-truth scoring (sources below).

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
fleet's real format). Phase 2 — DONE (see banner): rest of B5 (vector
composition family) + B6 (agent wiring) + drawing sets; B7 was scoped out
of the Phase-2 build and is now the TOP Phase-3 item (promoted to a
required leg by the SHX finding). Phase 3: B7 raster/OCR leg,
fill-cluster arrowhead representation (the Mecklenburg micro-dot
finding), B3 marks A/B, opportunistic B1 DXF-native LEADER/DIMENSION
ingest (would also let search_drawing_set use native entities at
confidence 1.0 on DXF input). Build on owner word; still worth 2-3
representative sheets from the owner's team as fixtures (scrubbed of
anything sensitive).

## GROUND-TRUTH HARVEST: COMPLETE (2026-09-04, commit 03e4ff5)

10 Mecklenburg DWG+PDF pairs live in
module_work/drawing_ground_truth/mecklenburg/ with per-sheet
native-entity truth JSON (21.01: 13 LEADER + 5 DIMENSION; 3001: 7+10;
10.31A: 5 MULTILEADER + 40 circles; TEXT everywhere). 179 pairs remain
available on the portal (MANIFEST.json documents the public API + URL
pattern — no browser needed for future pulls). DWG→DXF via ODA File
Converter 27.1 silent MSI (unsigned LibreDWG is blocked by Windows App
Control on this machine).

**FLEET-REALITY FINDING #2 (verified on all 10 real PDFs): zero text
layer.** Agency AutoCAD plots letter with SHX-stroked geometry — no
fonts embedded, nothing extractable as text, while vector linework
stays rich. Consequence: `find_text`/text-anchored queries on such
sheets require the B7 raster/OCR leg (render → OCR → map boxes to IR
coords) EVEN ON VECTOR PDFS; geometry-side composition is unaffected.
B7 is therefore promoted from "scanned sheets only" to a required leg
of the primary path. TrueType-font plots do carry a text layer — both
realities must be handled, and ingest should REPORT which kind of
sheet it sees (fonts/words present vs not) so the agent knows whether
text queries need OCR.

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
