# Drawing & submittal intelligence — design memo (2026-09-04)

> **SCOPE REFRAMED 2026-09-07 — READ THIS BEFORE THE REST OF THE MEMO.**
> Everything below concerns RECOGNIZING CAD constructs, and the owner has
> corrected that this was never the end goal: *"architects and engineers
> using an LLM to assist in reviews of design and construction documents...
> The ability to notice lines, arcs, callouts, dimensions, and area objects
> is important, but it isn't the end goal in itself."* And: an upload is
> usually a geotechnical REPORT — thirty pages of narrative, then tables,
> then figures — so *"the user [should] not have to distinguish between a
> language review and a visual review."*
>
> The construct work below remains correct and remains the foundation; it is
> a means. The current plan of record is
> `C:/Users/socon/.claude/plans/delightful-swinging-sky.md` — a review
> finding over a DOCUMENT, built as a vertical slice on *"what is the
> average spacing of the borings in this plan?"*. Owner decisions: PDF
> nearly always (so block/INSERT structure is not the mechanism), vertical
> slice before horizontal plumbing, and findings as DATA rather than a fixed
> deliverable. Surveyed verdict on that question today: it fails as a
> plausible wrong number rather than a refusal — no symbol census, no scale
> resolution, no spacing statistic, and `text_anchored_geometry` structurally
> cannot bind a label to a circle.

> **PHASE 3.1: COMPLETE + INDEPENDENTLY VERIFIED (ship-with-fixes,
> fixes applied 2026-09-05; planlens tip 7a611df, app 4505e46).**
> Verified real-sheet numbers: **leaders 21/25** (open-arrow
> representation: real plotters draw 3-vertex arrow outlines minus an
> edge, base+leg and chevron flavors), **dimensions 13/16** (split-shaft
> pairing, ends = CAD defpoints), worst-sheet FPs **44→11 (8 touching
> truth)**. Render-adjudicated precision is SHEET-CLASS dependent:
> curb-ramp "false" dims are mostly REAL drafted dims the native truth
> can't record (~14/15); note-heavy SHX sheets admit glyph junk at
> conf 1.0 (~5/14); leaders have NO precision story on lettering-heavy
> annotation-free sheets (letterform FPs — documented). Align guard:
> 0/160 random-anchor fits, all true fits intact (Poisson-significance
> + extent guard = the operative defenses). OCR corner-role fix: all 4
> /Rotate values within ~2 pt; rapidocr bounded <1.3. B1 DXF-native
> ingest live (conf-1.0 leaders/dims/attribs; planlens.dxf.truth =
> model-space corpus extractor). **Verifier's pre-existing find, fixed
> same day: render/snip's IR conversion was rotation-blind — the zoom
> loop mispointed on every /Rotate sheet; frame='ir' now does the full
> empirically-pinned conversion (bbox + marks), 13 pixel-check tests.**
> Deferred → Phase 3.2: flared two-stroke curved arrows (4 leader
> misses), witness-crossing verticals (3 dim misses), INSERT block
> explosion, live SoM A/B (owner-gated), layout blocks in
> planlens.dxf.truth.

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

## Phase 3.2 banner (2026-09-05) — recall closed, precision measured

**INDEPENDENTLY VERIFIED, SHIP-AS-IS (2026-09-05):** render-audit of
the default-confidence survivors found the detail-sheet "FP" counts
are MOSTLY real hand-drafted annotations native truth can't record,
plus a letterform tail (e.g. 2000b's 10 survivors at ~0.83 are junk —
confidence ranks within a sheet but is NOT calibrated across sheets).

**Close-out review, 2026-09-07.** A code review of the four unreleased
Phase-3.2 commits, and the fix round that followed, corrected three
things below. Every number in this banner has now been re-measured;
where a figure describes the committed tip `1f6551c` rather than the
close-out tree, it says so. The three sharp edges follow — (1) was
described wrongly and its deletion has been removed, (2) was a count
quoted without its threshold, (3) was a constant described in the
wrong form.

**(1) Crooked dimension arrows — CORRECTED 2026-09-07.** The earlier
wording here ("a dimension whose arrows sit >30 deg off-axis is
silently LOST entirely") was wrong in both its threshold and its leg,
and the deletion it described has been removed. Two independent tests
govern a dimension arrow, and they answer different questions:

- **direction** — does the arrow point OUTWARD along the line
  claiming it? The test is the signed intrinsic apex axis against a
  30 deg cone (`_ARROW_AXIS_MIN = cos 30`). Failing it is genuine
  CONTRADICTION: the geometry argues against the reading.
- **spine** — does the dimension line run INTO the arrowhead along
  the arrowhead's own spine? The tolerance is the candidate's own
  drawn half-width, so it introduces no constant and scales with the
  arrow. Failing it is WEAK EVIDENCE, not contradiction: the
  arrowhead may belong to some other construct this line merely
  grazes.

The continuous leg always graded these separately — off-spine falls
back to the shaft's own end (drawn geometry) and caps. The SPLIT leg
did not: it REFUSED to found a half-shaft at all when the arrow was
off-spine, so a split-shaft dimension with a crooked arrow vanished
with no proposal at any `min_confidence`. That is now the same
fallback-and-cap the continuous leg uses, with the same
`arrow_off_spine` evidence key — deliberately identical treatment of
an identical condition (grading them differently is what F15 exists
to remove).

The resulting ladder for a split-shaft dimension with a crooked arrow:
on-spine and inside the direction cone = full confidence; off-spine
but inside the cone = capped at 0.45 (`_UNCORROBORATED_CAP`); outside
the cone = capped at 0.25 (contradicted). 0.25 is reserved for actual
contradiction — if an implementation ever grades off-spine at 0.25,
that is a defect, not a policy. The remaining sharp edge is a
VISIBILITY one, not a loss: the middle case sits below the 0.5 default,
so a default call does not show it and the caller must lower
`min_confidence` to 0.45 to surface it.

A FOURTH sharp edge, measured on the round-4 close-out tree and new
with it. The CONTINUOUS leg resolves an off-spine end differently from
the split leg on purpose — it publishes the shaft's own end, which is
drawn fact, rather than the projected apex. That is right when the
arrows sit inside the shaft pointing outward, and it under-reports in
the arrows-OUTSIDE style, where the shaft stops at the arrow bases: a
witnessed, texted 120 pt dimension whose arrow is drafted past
asin((base/2) / h) = 9.59 deg reports its end at the shaft tip 112.8
rather than the defpoint 120.0, one arrow length short, and is capped
at 0.45 so it is not called. Nothing is lost — it is still proposed at
`min_confidence=0.0` — and the arrowhead steal that this replaced was
far worse: before the fix that same construct claimed a NEIGHBOURING
REAL LEADER's arrowhead at 0.964 and `exclude_dimensions` then deleted
that leader at every threshold.

The evidence that would settle it is already in the scene: a witness
line stands at the true defpoint, and `_witness_at` is already called
at each published end. Using it means resolving the ends BEFORE the
confidence hoist that computes the cap ceiling ahead of the witness
search — the hoist that took sheet 3001's dims@0.5 from 8.5 s to
0.4 s. That is the next refinement, and it is a real one; it was not
attempted in round 4 because reordering a measured hot path at the end
of a four-round remediation is how the last three regressions were
born. Pinned meanwhile by
`planlens/ir/tests/test_text_bearing_scenes.py::TestSkewedDimensionDoesNotStealANeighbouringLeader`,
which asserts both halves: the leader survives at every skew from 0 to
25 deg, and the residual is exactly the shaft-tip fallback described
here.

One hard refusal survives on the split leg and is deliberate: the
arrowhead's centroid must sit PAST the shaft tip along the terminal
direction. That is a *kind* judgment, not a *quality* judgment — an
arrowhead sitting inside the half-shafts is the arrows-INWARD
arrangement, which is the continuous leg's construct and which the
continuous leg proposes. The refusal is a routing decision, defensible
only because the construct has a home elsewhere; it depends on the
continuous leg actually picking that arrangement up, and if that is
ever measured false the refusal must become a cap like the rest.

**(2) "Dim proposals zero at 0.3+ on pure-notes sheets" was two
claims, and both needed narrowing.** Only 2000a/2000b/3000 are
genuinely pure-notes sheets. 10.17a and 11.01 are DETAIL sheets whose
annotation was hand-drafted rather than placed as native DIMENSION
entities — they are not "no annotation", they are "no annotation
native truth can record" — and they are covered by the ~14/15-real
render-adjudication caveat.

The counts, re-measured 2026-09-07 rather than copied: at the
committed tip `1f6551c`, 10.17a and 11.01 emit **17 and 18** dim
proposals at the 0.5 default and **17 and 28** at 0.3. So the
"17-18" printed here before was right — but only at the DEFAULT, and
it was filed under a claim about the 0.3 threshold, where the true
tip figures are 17 and 28. That mismatch is the whole lesson: a
proposal count is meaningless without the threshold it was taken at.

Separately, a zero count at a LOW threshold is a fragile thing to
promise at all. Capping rather than deleting is what puts an
uncorroborated construct in the 0.25-0.45 band, so every cap added to
the pipeline RAISES low-threshold counts by design — the Phase-3.2
close-out round takes 10.17a from 17 to 44 dim proposals at 0.3 while
its 0.5 count stays at 17. Quote default-confidence counts; treat the
0.3 observational band as the capped tier, not as false positives.

**(3) The 0.75x detach bound is not 0.75x of one fixed scale.** Both
documents previously described it as 0.75x the sheet-statistic
arrowhead scale; the code takes 0.75x the LARGER of that scale and the
candidate's OWN axial length (`|apex - base|`). The physics: a drawn
arrow's centroid sits ~2/3 of its own length behind its apex, and the
candidate gate deliberately admits shapes up to 1.5x the sheet
estimate (the estimate ran ~20% under the real plotted arrows on one
validation sheet), so a bound tied only to the sheet statistic makes
the biggest genuine arrowheads unable to attach to anything. The old
"~23% margin over the measured true population" reading described the
sheet-statistic form only.

Built in planlens (commits 2b4eee1..1f6551c), scored against the
committed Mecklenburg corpus. **Leader tips 25/25 and dimension
defpoints 16/16** at the scorer's 0.3 threshold (23/25 at the 0.5
default — two sparse-dot tips live at the 0.45 cap).

What actually moved the numbers (the plan's flared-two-stroke
arrowhead theory did NOT survive measurement — every missed tip's
arrow was already a candidate):

1. **Signed arrow-direction attach** (`_dim_arrow_attach`): a
   dimension arrow must point OUTWARD along its line within 30 deg of
   its intrinsic apex axis. The old fold-blind best-vertex alignment
   can't score below ~cos30 for any triangle vs any crossing line, so
   false dimensions were claiming true leaders' arrowheads and
   `exclude_dimensions` silently dropped the leaders — all 4 residual
   leader misses. Measured separation: true dim arrows +1.000,
   impostors -0.998..+0.208.
2. **Short both-arrowed continuous dims**: 3001's 'T='/'X=' and
   21.01's chain dim are narrow constructs (9.7-15.2 pt shafts between
   two outward arrows) that the min-shaft-length floor rejected — all
   3 residual dim misses. Continuous proposal ends are now the arrow
   APEXES (CAD defpoints; recovered dims match at ~0.0 pt).
3. **Letterform precision caps** (cap to 0.45, never delete): arrow
   must point along its shaft (signed axis) AND the shaft must END at
   the arrowhead — within 0.75x the LARGER of the sheet arrowhead
   scale and the candidate's own axial length (see sharp edge (3);
   measured true 2.2-5.4 pt vs junk p50 9.6 pt). Worst notes sheet:
   295 -> 2 leader proposals at default confidence (all seven
   zero-annotation sheets in the scorer ledger); corpus recall
   unchanged. Rejected candidate vetos, with measurements:
   small-stroke density and same-scale-neighbor counts both fail BOTH
   ways (real arrows sit inside stipple at n up to 1155; big isolated
   title letters have n=0-1).
   *The under-reported half of item 3: dimension PRECISION improved
   too*, at unchanged recall (re-measured 2026-09-07, committed tip
   `1f6551c` vs the close-out round). Dimension proposals at the 0.5
   default fell on every sheet carrying native dims — 21.01 19->5,
   3001 15->10, 11.01 18->12, 10.31A 15->10 — and what went away were
   the FALSE ones: 21.01 and 3001 now return exactly their native
   dimensions and nothing else (5/5 and 10/10, zero unmatched) where
   the tip returned 14 and 5 unmatched alongside. Defpoint recall is
   **16/16 at the 0.5 DEFAULT**, not merely at the 0.3 observational
   threshold — a stronger statement than this memo has been making.
   Same cause as the leader caps: signed attachment stops an
   arrowhead founding a dimension it does not point along. Nobody
   wrote this down at the time; it is the strongest precision
   evidence the phase produced.
4. **DXF INSERT explosion** (`from_dxf explode_blocks=True` default):
   block geometry ingests with exact transforms + `block:<name>`
   provenance, depth-8/50k-entity caps (10.17a 227->788 entities,
   5003 170->1870); native annotation counts unchanged. **Layers are
   now RESOLVED, not copied**: layer `"0"` inside a block definition
   means "inherit from the INSERT" in DXF, so exploded entities take
   the INSERT's layer — user-visible, and it moves entities between
   layers. Measured off the corpus DXFs (ezdxf `virtual_entities`,
   2026-09-07): 557 entities re-homed on 10.17a, 158 on 11.01, 1696
   on 5003, with the distinct layers CARRYING GEOMETRY going 3 -> 2 on
   10.17a and 5003 (11.01 stays 4); the other seven sheets have no
   layer-`"0"` block geometry. The `n_layers` METADATA field counts
   something else — every layer name seen during ingest — and reads
   4, 8 and 3 on those three sheets. Any caller that hard-codes a
   layer name or counts layers sees different numbers than before.
5. **Ledger items**: `planlens.dxf.truth` now extracts paper-space
   layouts; the corpus truth files were regenerated (additive attribs
   field) and verify 10/10 byte-identical. SoM A/B verdict still
   owner-gated (som_ab_check.py not run).

Residual, documented per tip: the two 21.01/3001 sparse-dot tips
surface only at 0.45 (below-default; capped junk-luck proposals near
the tips carry the 0.3-threshold match). Detail-sheet survivors at
default confidence may be REAL manual annotations native truth cannot
record — the standing render-verification caveat applies.

**The corpus's structural blind spot — read before quoting a corpus
number as evidence.** All ten Mecklenburg sheets are SHX-stroked with
NO text layer (`has_text` false on 10/10, verified 2026-09-07). The
text-corroboration channel of every construct score is therefore
identically ZERO on this corpus, which makes it blind BY CONSTRUCTION
to every behavior gated on text presence, `ext_ends`, or witness
corroboration. "The corpus did not move" is not evidence that a
text-dependent change is safe — it is the expected reading whether the
change is safe or not. Text-dependent behavior must be measured on a
text-bearing synthetic scene (`planlens.testing`), and that scene must
be left behind as a permanent fixture. This is how a fix that dropped
a real leader on any ordinary text-bearing drawing measured "zero
change" here.

Same category, smaller: the **blunt-terminator** branch (box / dot /
oblique-blob dimension ticks — no pointing direction, so they take the
sign-blind path) is validated by synthetic fixtures. On real sheets it
fires only into the CAPPED band, never above the call threshold
(re-measured 2026-09-07 on the close-out tree): across all ten sheets
**19** proposals carry `blunt_terminators` evidence at
`min_confidence=0.0`, **17** at 0.3 and **none at the 0.5 default** —
21.01 contributing 17 / 15 / 0, 11.01 contributing 2 / 2 / 0, and the
other eight sheets none at any threshold. So no default-confidence
corpus result rests on that branch, and its behavior above the call
threshold is unmeasured on real drafting.
