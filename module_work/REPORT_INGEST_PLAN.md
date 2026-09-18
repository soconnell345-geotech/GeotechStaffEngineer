# Report ingest — geotechnical report → organised record + summary + DIGGS 2.6

**Status: WP0–WP5 BUILT; 5.20.0 RELEASED with planlens 0.6.0 (2026-09-17);
cluster scoring pending the owner's run.** WP0 and WP1b shipped in app 5.19.0
with planlens 0.5.0. WP2 through WP4 — the record, the log reader and DIGGS
2.6, the lab reader, the narrative reader and the reconciler, the writers,
the deterministic graph, the folder runner and the app sub-agent — plus the
WP5 vision-first label experiment (`report_ingest/vision_labels.py`, cluster
stage `vision_labels`) shipped as 5.20.0 with planlens 0.6.0 (tags `v5.20.0`
and `v0.6.0`). The calc reader is the one WP5 item still parked.

**Every model number for WP2–WP4 is still missing, and that is the state.**
What the ledger holds for the three readers is the baseline WITHOUT a model:
`log_grid` alone for the logs, the page's own detected tables alone for the
lab sheets, and nothing yet for the narrative. The development-engine
checkpoints were never run for any of the three. The numbers arrive from the
owner's four-stage `score_on_cluster` run, on the tier that will do the work
— which is why the sub-agent ships with `enable_report_ingest=False` and why
no reader's accuracy is quoted anywhere. WP5 is the one package not built.

(Started 2026-09-16, owner: "run with what you have".) Drafted 2026-09-16 after reviewing the report corpus (38 reports,
7,829 pages) with planlens 0.4.0; revised the same day with the owner's two
query schemas, the Azure Document Intelligence (DI) results found in the
private repo, and the owner's steer that neither the old page labels nor the
pipeline shape below are fixed.

**End goal (owner's words):** well-organised data about each geotech report
that can be used in other workflows, either as a sub-agent in the primary app
or as a dedicated WikiLLM-style report-library agent. The **record** is the
product; the summary page and the DIGGS file are two exports of it.

**Privacy rule for this document.** This repo is PUBLIC. The corpus lives in
a gitignored folder (`module_work/field_feedback/2026-09-16_report-ingest/raw/corpus/`,
manifest `MANIFEST.md` there) and reports are referred to here only by ID
(R01–R38). The public-source reports may be named by firm; the rest may not be
named, located or quoted. The old attempt is the owner's private
`GeotechReportExtraction` repo; it is the corpus home and otherwise not used.

---

## In plain English

We want the app to take a geotechnical report PDF and hand back one
well-organised record of what is in it: the report's identity and scope, the
answers to the standing questions the owner has been asking of these reports
for years (who wrote it, how many borings, what foundations were recommended,
what natural hazards, which seismic code, and so on), the boring logs and lab
tests as data, and a note of what could not be read and why. From that record
we write a one-page summary an engineer can read, a DIGGS file our existing
plots and tools already understand, and a page in the same style as the
owner's WikiLLM library so a report-library agent can search hundreds of them.

The last attempt trained a page classifier and wrote a hand-tuned parser for
each firm's log template. Every new firm meant a new parser, and lab tests
were never done. The difference now is planlens: it already cuts a report into
its parts (the narrative, each appendix, each boring log, each lab sheet) from
the document's own headers and dividers, and gives every piece of text with
its position on the page. Tested today on five reports from five firms, it
found the narrative, the appendix dividers, every boring log and lab sheet and
the calc printouts without training. On a boring log it hands back the depth
scale and the column headers as located text, so values can be put in the
right column at the right depth by geometry rather than by guessing. For
scanned pages, Azure DI gives the same located text plus tables, and planlens
already reads a DI result; a free on-machine OCR is the fallback.

So the plan is: planlens (and DI for scans) does the "where", small focused AI
readers do the "what" (one boring log at a time, one lab sheet at a time, the
narrative once against the owner's question list), a deterministic checker
joins it all up and writes the outputs, and every number carries the page it
came from so a reviewer can check it. Each stage is scored against hand-checked
answers before the next starts. The 4,300 pages the owner labelled by hand
are the scorecard for the page-sorting step; I hand-truth the logs and lab
tests from the rendered pages.

Two planlens bugs turned up during the review and get fixed first: it calls
different scanned lab sheets "the same page" when they share a template and a
DRAFT watermark, and it does not warn when a page's text layer is garbage.

---

## 1. What the corpus looks like (measured 2026-09-16)

Sweep of all 38 PDFs with PyMuPDF; page map and segmentation with planlens
0.4.0 on five of them. Full per-report table in the gitignored manifest.

| Fact | Number |
|---|---|
| Reports / pages | 38 / 7,829 |
| Median report length | ~150 pages (range 15–729) |
| Reports with a usable text layer on ≥90 % of pages | 25 |
| Reports that are pure image scans with NO text at all | 2 (R13, R38) |
| Reports mostly image-only | 1 (R19: 252 of 275 pages) |
| Reports with OCR text laid over scanned images (1990s–2000s) | 6 (R09–R12, R14, R35) |
| Reports with a large scanned appendix inside a text report | 4 (R15: 75 pages; R03: 79; R33: 38; R23: 54) |
| Page sizes | letter dominates; tabloid figures; A4 lab sheets from local labs; a few D-size and long plots (up to 23×69 in) |
| Hand-labelled page-type ground truth | 4,300 pages, 15 reports, 23 labels (lab testing 1,092; calculation 1,009; appended report 494; narrative 439; boring log 281; test pit log 251; photos 132; …) |
| Azure DI results already on hand | **21 of the 38 corpus reports** (private repo `sample_reports/Reports_DI`, copied gzipped into `raw/di/<ID>.json.gz`, 84 MB); one more result is truncated (R17) and one has no PDF; a further 22 results for a different batch (~2.4 GB) stay in the private repo. Covered: R13 (pure scan), R19, R15's scanned appendix, R23, all six OCR-over-scan reports. Not covered: the public reports, R38, R31, R32, R33, R34, R35 |
| Languages seen in appendices | English; French lab sheets (R28); Spanish (R25) |

What planlens already gives on these reports, with no report-specific code:

- **Segments = the report's parts.** R36 (Terracon, 94 pp, public): narrative
  as one segment (pp 2–25, running header), appendix dividers, boring logs as
  a `form` run headed "BORING LOG NO. B-1 … Page 1 of 3", CPT data, lab sheets
  by title ("ATTERBERG LIMITS RESULTS", "SWELL CONSOLIDATION TEST | ASTM
  D4546", "DIRECT SHEAR TEST | ASTM D3080"), a corrosivity report, a USGS
  design-maps printout, pile calc pages. 1.7 s.
- R04 (Langan, 97 pp, public): figures, DCP data forms, test pit logs, lab
  forms, photo appendix, each its own segment. R37 (KIM, 48 pp, public):
  narrative, plan, five "BORING NUMBER SB-1" pages, infiltration tests, lab
  appendix.
- R15 (2023, 202 pp) and R28 (2023, 455 pp), private: narrative with printed
  page numbers, figures, boring and test-pit logs headed "TEST BORING …
  Boring Number: SB-01", sieve/hydrometer sheets, a 74-page scanned lab
  appendix, French-language lab sheets, calc printouts (settlement, pile
  capacity). 8–9 s each.
- **Located text on a boring log is a coordinate system.** On R36 p35 the
  depth-ruler ticks sit in one x-band; the rotated column headers ("DEPTH
  (Ft.)", "WATER CONTENT (%)", "DRY UNIT WEIGHT (pcf)", "ATTERBERG LIMITS
  LL-PL-PI", "PERCENT FINES") each occupy a known x-band; the values
  ("5-9-12", "N=6", "13", "116", "31-20-11", "52") sit under them; layer
  descriptions sit in their own band with the top-depth tick beside each.
  planlens already flags such pages ("ruled form … view the page to read the
  layout").
- `evidence.unmapped_chars` already counts glyphs with no Unicode mapping
  (R28 calc printouts: 1,372 of 2,929 characters are garbage).
- **Azure DI is already a planlens text source** (`planlens.document.azure_di`):
  `pages_needing_ocr(doc)` lists the pages worth paying for,
  `AzureLayout.analyze(fh_doc.analyze_layout, pdf_bytes, pages=…)` runs it
  through the Funhouse SDK on the cluster, and `open_document(pdf,
  text_source=layout)` puts DI's lines, paragraph roles (title, section
  heading, header, footer) and tables into the same displayed-page frame as
  everything else. Nothing new to build for the scan path except the
  harness. **Checked on R13 (the 2005 pure scan, 197 pages):** with its DI
  result attached, every page has text, 59 segments come out (the narrative
  as one run, figures, each boring log as its own segment headed by the log
  title), and a log page yields the header fields as located lines (boring
  number, drilling method, hammer type, ground elevation, the groundwater
  table) plus a DI table of the log body with depth, description, USCS
  symbol, elevation, sample IDs and blow counts. 7 s. DI marks sample
  symbols as checkbox tokens; the reader prompt must know that.

Two defects to fix before anything is built on top:

1. **Image-duplicate over-claim on scans.** R15 pp 112–184 (69 pages) are
   reported as `duplicate_of` p111 (`duplicate_rule: image`). Rendered, they
   are different lab sheets (three sieve/Atterberg forms for different
   samples and a density table): shared template + DRAFT watermark + little
   ink. An ingest that skipped "duplicates" would drop the whole lab appendix.
2. **No text-reliability flag.** `unmapped_chars` is reported but nothing
   consumes it; a page whose text layer is mostly unmapped must be treated
   like a scan (DI, OCR or vision), not read as text.

---

## 2. The record (the product) and its three exports

`report.record.json`, one per report, schema in `report_ingest/model.py`
(pydantic, JSON-schema exported for the WikiLLM-style agent). Sections:

| Section | Content | Filled by |
|---|---|---|
| `document` | file identity, page count, page kinds, text/scan mix, planlens and DI versions, run cost | planlens, deterministic |
| `general` | the owner's **general query schema** (§3), field names kept verbatim | narrative reader |
| `natural_hazards` | the owner's **NHA query schema** (§3), field names kept verbatim | narrative reader |
| `investigations` | borings, test pits, CPTs, DCPs: location, depths, layers, samples, SPT, water levels, drilling details | log reader |
| `lab_tests` | typed results per test kind, linked to investigation + sample + depth | lab reader |
| `calcs` | (later) each calc printout: method, program, key inputs, key results | calc reader |
| `qa` | what was extracted, partial, skipped or in conflict, and why; cross-checks (narrative counts vs logs found, summary table vs sheet values) | reconciler |

Every value-bearing field carries `Provenance(page, bbox, method:
text|di|ocr|grid|vision, confidence, note)`.

Exports written next to it:

| Export | Form | Consumer |
|---|---|---|
| `report.summary.md` | The `general` and `natural_hazards` answers in prose with page citations, then counts of what was extracted and what was not | The user; the chat reply |
| `report.page.md` (+ one row in a SQLite index) | The owner's WikiLLM page format (title, authors, year, source, doc_type, disciplines/topics/methods tags, summary, key takeaways, key parameters, status, confidence) with the record sections rendered beneath | A dedicated report-library agent; search across many reports |
| `report.diggs.xml` | DIGGS 2.6, XSD-valid (pydiggs, bundled schema, offline) and round-trip-equal through the app's own `parse_diggs` | `subsurface_characterization` plots and every tool that takes `site_key` |

The compact tool result the primary agent sees is counts, the paths and the
summary's first lines, never the bulk payload.

---

## 3. Narrative facts: the owner's two schemas are the working list

Kept verbatim as field names so past query outputs stay comparable. Two
sections: **general** (documentType, quickSummary, postName, propertyType,
projectNumber, projectName, projectPhase, primeContractor, primeAe,
geotechnicalEngineerFirm, testingProgramSummary, boringCount, testPitCount,
cptCount, tableCount, figureCount, previousInvestigationCount, strata,
structureCount, structureList, outsideProject, boringDictionary,
testPitDictionary, recommendedFoundations, bearingCapacity) and
**natural_hazards** (liquefactionPotential, asceSevenVersion,
earthHazardsExposed, seismicCodeUsed, geophysicalTestingMention,
soilCorrosion, siteResponseMention, hazardAnalysisMention, siteClass,
seismicParameterSummary, naturalHazardSummary, reportDate). The enumerations
(documentType, propertyType, projectPhase, liquefactionPotential,
earthHazardsExposed) are kept as written.

Where the old string fields hid data, the record adds a typed twin beside the
original string, never instead of it:

| Field | Typed twin |
|---|---|
| `bearingCapacity` (strings) | list of {value, unit, foundation type, condition (e.g. "on engineered fill"), citation} |
| `strata` (a "dictionary" in one string) | list of {name, description, top/bottom depth or elevation, USCS, citation} |
| `boringCount`, `testPitCount`, `cptCount`, `boringDictionary`, `testPitDictionary` | as stated in the narrative, **plus** the counts and IDs actually found in the appendix; a mismatch is a QA entry, not an error |
| `tableCount`, `figureCount` | counted deterministically from "Table N" / "Figure N" captions in the main body; the model's answer is checked against it |
| `*Mention` fields, `soilCorrosion`, `outsideProject` | {answer: yes/no/unclear, citation} |
| `siteClass`, `asceSevenVersion`, `seismicCodeUsed` | normalised value + citation |
| `reportDate` | ISO date (the schema's description is a copy-paste of naturalHazardSummary's; the field is the report date) |

Every answer carries page citations from `read_document`. The list is open:
new questions are new fields in `model.py` plus one line in the reader
prompt, and the record versions itself.

---

## 4. Architecture (a starting shape, not a contract)

Principle carried over from planlens: **geometry says WHERE, the model says
WHAT.** Orchestration is deterministic Python; language models read one
bounded thing at a time and return typed JSON; every number carries the page
and box it came from; nothing is asserted without evidence. The fixed points
are those three sentences and the measured gates in §6. How many readers
there are, and what each reads, will follow the measurements; the split
below is only where I would start.

```
report_ingest (CompiledSubAgent: its own deepagents graph, one primary tool)
|
+- 0. planlens: open -> page_map -> segments -> page labels (new)     deterministic
|       (+ DI text source for pages that need it, when a DI result exists)
|       a label per page: narrative, figure, plan, profile, boring_log,
|       test_pit_log, cpt_log, dcp_log, lab_test, calculation,
|       appended_report, photos, divider, cover/toc, other
|       (+ evidence, + text_reliable)
|
+- 1. work items                                                     deterministic
|       narrative: the narrative segment(s)
|       logs: one per log (segment run headed by the log title,
|             "Page 1 of 3" continuations folded in)
|       lab: one per lab sheet or per multi-page test
|       calcs: one per printout (later)
|
+- 2. readers (sub-subagents; one item per call; typed JSON out)     LLM + tools
|       narrative_reader   read_document on narrative pages; fills
|                          general + natural_hazards with citations
|       log_reader         log_grid rows (new planlens helper) + rendered
|                          page -> one Investigation
|       lab_reader         located text (+ DI table cells on scans) +
|                          rendered page -> one LabTest; digitises a
|                          curve only when the values are not tabulated
|       calc_reader        (later) method + key inputs/results, prose only
|
+- 3. reconciler                                          deterministic (+LLM for conflicts)
|       joins lab -> investigation/sample by ID + depth; narrative
|       counts vs logs found; summary table vs sheet values; unit
|       normalisation; the QA section
|
+- 4. writers                                                        deterministic
        record.json -> summary.md, page.md (+ SQLite row), diggs.xml
        (pydiggs XSD gate, parse_diggs round-trip gate)
```

### Two model passes before any reader runs (owner direction, 2026-09-16)

The rules in step 0 are a first draft of the page labels, never the last
word. The owner's point stands: a page only makes sense in the context of
the rest of the report, and near-perfect labels on the key content
(narrative, location plan, subsurface profiles, logs, calcs, lab data)
matter more than tokens. So two model passes sit between step 0 and the
readers, both with tools and a generous budget:

- **0b. Document triage.** One call over the whole-document ledger (one
  line per page: kind, rule label + confidence + the rule that fired,
  heading, running header/footer, printed page number, segment, text
  reliability, DI used), the cover and table-of-contents text, and the
  first contact sheet. It answers: what kind of document is this (the
  owner's `documentType` enumeration: full report, addendum, appendix or
  figures only, partial file, other); is it one document or several bound
  together (volumes, appended prior reports, a data report inside a design
  report); what is missing (no narrative, no logs, no lab); scan and
  unreliable-text fractions; languages; whether page numbering restarts and
  whether the table of contents matches what was found. It emits a
  `document_profile` and a **workflow** choice: standard, appendix-only,
  partial, multi-document, scanned, or needs-a-person. The readers that
  follow are chosen by that workflow, and an odd report is flagged instead
  of being forced through the standard path.
- **0c. Label review.** An agent loop, not a single call, over the same
  ledger plus everything the report says about itself: the table of
  contents and its lists of figures, tables and appendices; every divider
  and fly-sheet's text; captions of figure pages; the section headings the
  narrative carries. It checks the rule labels against that context (the
  figure list says Figure 2 is the boring location plan, so the page whose
  caption reads Figure 2 is `plan`; Appendix C is "Laboratory Test Results",
  so a `form` page inside it that the rules left as `other` is almost
  certainly `lab_test`), spot-checks pages by rendering them, walks the
  contact sheets as a gut check, and returns corrected labels with a reason
  for each change, plus the structure it reconciled (TOC section → page
  range as found). Only then are work items built.

Both passes are scored the same way as the rules: rules-only versus
rules-plus-review, on the in-sample, held-out and out-of-sample sets. The
target after review is ≥ 0.98 recall and precision on the key-content
labels (narrative, plan, profile, boring_log, test_pit_log, cpt_log,
dcp_log, lab_test, calculation). Token cost is recorded, not optimised.

Why a deterministic loop rather than letting the sub-agent's model plan the
fan-out: it is budgetable (one call per log page, one per lab sheet),
testable offline with recorded reader outputs, restartable, and it cannot
forget an appendix. deepagents 0.6.8 supports this shape: a
`CompiledSubAgent` is any runnable with a `messages` key, and its
`structured_response` becomes the tool result the primary sees. The same
graph runs headless over a folder of PDFs for the report-library use.

### Text sources, in order of preference per page

| Page situation | Source | Notes |
|---|---|---|
| Text layer present and reliable | PDF text via planlens | Free, exact boxes |
| Scanned, or text layer unreliable, DI result available | **Azure DI** through planlens `AzureLayout` | Results for 21 corpus reports are already in `raw/di/`; for new reports the owner runs DI on the cluster (Funhouse SDK); gives lines, heading roles and tables with header cells on scans; only the pages `pages_needing_ocr` names need be sent; cost is a few cents per report at the known rate |
| Same, no DI result | **RapidOCR** via planlens `[ocr]` (`rapidocr-onnxruntime`, models inside the wheel) | Free, offline; installed in the dev venv; needs the same Nexus install check on the cluster that rapidfuzz had. Not related to rapidfuzz (which does fuzzy text matching) |
| Neither reads it | The page image to the reader model | Last resort; every such page is a QA entry |
| Any page | The page image to a cheap vision model with structured output (experiment, scored in 5.20.0) | Not a text source at all — it skips the text and asks what the page IS. `report_ingest/vision_labels.py`, scored against the same hand labels as the rules by `score_on_cluster(stages=("vision_labels",))` on `funhouse-gpt-low`, in one call per page, one per six-page contact sheet, or (5.21.0) one per **window of 36 stamped full-size pages with contact sheets of the whole report beside them** — `mode="document"`, which exists because page mode lost on `plan` and `lab_test`, the two labels settled by knowing which appendix a page is in; the binding limit is the endpoint's measured 50-image-per-request cap, not the context window |

Locally (this machine) DI cannot be called, so the harness accepts DI JSON
dropped into `raw/di/` by the owner after a cluster run; RapidOCR covers the
rest locally.

### The old XGBoost page classifier

Its 134 features are regex counts and TF-IDF over page **text** (only two are
DI-specific), so it can be scored on planlens text with no DI files. It gets
one job: a **benchmark** on the same 4,300-page scorecard as the planlens
page labels (WP1). If the rules match or beat it, it is not adopted; if it
wins on some label, it becomes a tie-breaker for that label. It is never a
dependency of the ingest (xgboost + scikit-learn would need Nexus clearance
and add nothing the rules cannot be taught).

### Where each piece lives

| Piece | Home | Why |
|---|---|---|
| dup fix, `text_reliable`, page labels, `log_grid`, OCR wiring | **planlens** (public, MIT) | Generic document facts; useful to any MCP host; measured on the corpus but never carrying corpus content |
| record model, reader prompts and schemas, reconciler, writers, `report_ingest` graph and tool, library indexer | **app**, new package `report_ingest/` | Geotech semantics, the owner's schemas, DIGGS, deepagents wiring |
| corpus, DI results, ground truth, measurements | `module_work/field_feedback/2026-09-16_report-ingest/` (`raw/` gitignored) | Privacy |

---

## 5. Work packages, in order

Each package ends with a measurement on the corpus and a commit. Nothing is
released until the owner says so. One Opus 5 builder sub-agent at a time,
Fable leads and reviews; suites run in the foreground. Hand-truth is mine,
from rendered pages, kept in the private ledger for the owner to spot-check.

### WP0: planlens fixes + corpus harness (small)

- **Dup fix.** On `scanned` pages require the image-hash match AND either a
  text-hash match or a tighter distance, and withhold the claim when the
  page's ink fraction is low or a large diagonal watermark is present
  (measure ink fraction and hash distance on R15 pp 111–184 first).
  Acceptance: 0 of the 69 R15 pages claimed; the synthetic positive still
  found; the corpus numbers the README publishes unchanged.
- **`text_reliable`** on `PageSummary` from `unmapped_chars / text_chars`
  (threshold measured on R28's calc printouts vs its clean pages); when
  false, `needs_ocr` is set and `read_document` says so.
- **Corpus harness** (app repo, gitignored data): loader for the 38 PDFs by
  ID; loader for the 4,300 hand labels mapped onto the page labels (the
  mapping is a table in the harness, easy to change); DI JSON loader
  (`raw/di/<ID>.json` → `AzureLayout`); one `measure_*.py` per work package
  that prints precision and recall and appends to `MEASUREMENTS.md`.
- **DI results on hand** for 21 reports (`raw/di/`), so scans are in scope
  from WP1 on. Not blocking, when convenient: DI for R38 and the scanned
  pages of R03 and R33 (the harness prints the page ranges with
  `pages_to_azure_range`), and a fresh export of R17's truncated result.

### WP1: page labels in planlens, scored against 4,300 hand labels

- Rules over evidence planlens already has: segment title, running header
  and first-heading keywords (multilingual: boring, sondage, sondeo, log,
  test pit, CPT, DCP, sieve, hydrometer, Atterberg, consolidation, triaxial,
  direct shear, unconfined, proctor, CBR, corrosivity, résistivité, …);
  divider pages ("APPENDIX B", "ANNEXE"); page kind; printed-page continuity
  ("Page 2 of 3" folds into the previous log); calc-printout signatures
  (program banners, monospace numeric dumps); appended-report detection (a
  nested document with its own cover and numbering, which planlens already
  finds as a nested segment); DI paragraph roles when present.
- Output: a label plus evidence per page, and the **work items** (§4 step 1)
  with page ranges and the log or test title.
- Acceptance: recall and precision ≥ 0.90 on boring_log, test_pit_log,
  lab_test, narrative and calculation over the 15 labelled reports; every
  miss listed in the ledger. The XGBoost benchmark is scored on the same
  table. Other labels are reported, not gated.
- Shipped as a planlens tool (`document_labels`) so an MCP host gets it too.

### WP1b: document triage + label review (the model passes of §4)

- **Inputs planlens must provide** (added to WP1's deliverable): the
  per-page ledger line; `document_outline(doc)` = table-of-contents
  entries with their printed page numbers, the lists of figures, tables and
  appendices, every divider or fly-sheet with its text, the caption line of
  each figure-kind page, and the narrative's section headings; contact
  sheets on demand.
- **Triage** (`report_ingest/triage.py`): one structured call →
  `document_profile` (fields in §4 0b) + `workflow`. Scored on all 38
  reports against my hand verdict per report (kind, completeness,
  bound-together documents, scan fraction, TOC agreement).
- **Label review** (`report_ingest/label_review.py`): an agent loop with
  three tools (`read_page`, `render_page`, `contact_sheet`) and a spot-check
  budget that scales with page count; output = final labels + reasons +
  reconciled structure. Scored rules-only vs after-review on the in-sample,
  held-out and out-of-sample sets; every change the review makes is logged
  so a wrong "correction" is visible.
- Models (owner correction 2026-09-17): the system runs in Funhouse through
  the OpenAI Prompter API, so **scoring happens on the cluster through
  Prompter, run by the owner from a notebook cell**
  (`report_ingest.cluster_scoring.score_on_cluster`, test wheels of both
  branches uploaded by the owner, corpus in a Volume or synced SharePoint
  folder, results written to `/tmp` or a Volume and brought back to the
  ledger). Offline tests use fake engines. A Claude dev engine exists for
  prompt iteration only; its numbers are never quoted as the system's
  accuracy. A first checkpoint on six reports with that dev engine (about
  $1 and 2.5 min per report; 0.889 → 0.939 strict on 677 pages) is
  recorded in the ledger as indicative only.
- Gate for WP2: ≥ 0.98 P and R on the key-content labels after review on
  the held-out and out-of-sample sets, and the triage verdict right on every
  report whose structure is unusual (multi-volume, appendix-only, partial,
  scanned, appended prior reports).
- **Before any release of this package:** `pyproject.toml` must add
  `report_ingest*` to `[tool.setuptools.packages.find]` (its include-list
  omits it today, so a release wheel would ship WITHOUT the package and fail
  on the cluster as an import error) and `report_ingest` to pytest
  `testpaths` (98 tests otherwise never run in the gate). The test wheels
  in `dev/v5_test_wheel/report_ingest_test/` inject that line into a
  throwaway copy; a release must not rely on that. Also known: on Prompter
  a tool result cannot carry an image, so rendered pages ride in a user
  message after the tool messages; the review does slightly different work
  on the two engines and the README says so.

### WP2: boring logs → investigations → DIGGS borings (the core)

- **`log_grid(page)` in planlens**, a generic form reader with no firm
  templates: (1) column headers = text lines of any rotation in the header
  band, merged into x-bands; (2) depth ruler = a monotone numeric series in
  one x-band with regular y spacing → a linear y→depth map (plus the
  elevation ruler when present); (3) every remaining text line → (column,
  depth at y); (4) description band → layers with the top depth from the
  tick beside them or from the stratum line; (5) rows plus a confidence per
  row (ruler-fit residual, header match). Works the same on DI/OCR lines.
  Output is data with boxes, never a claim.
- **`log_reader`** (app): input = `log_grid` rows for all pages of one log
  plus the page image(s) at ~110 dpi; output = one `Investigation`. Prompt
  rules: use the rows; use the image only to resolve what the rows leave
  ambiguous (sample symbols, water-level symbols, refusal notation such as
  50/3"); never invent a depth; record hammer type and energy when printed.
- **DIGGS writer** for Project, Borehole, SamplingActivity/Sample,
  DrivenPenetrationTest (SPT), lithology observations, WaterLevelObservation,
  and the field values on the log (moisture, unit weight, Atterberg, fines,
  qu) as their tests. Elements confirmed present in the bundled 2.6 XSD.
  Gate = pydiggs schema check plus `parse_diggs` round-trip equality.
- **Ground truth (mine):** 12 logs across ≥ 6 templates (R36, R04, R37,
  R15, R28, R06, R01, and one 1990s OCR-over-scan report), each transcribed
  from the rendered page into the record format and kept in the private
  ledger. Metrics: N-value exact; sample depth within 0.15 m; layer top
  within 0.3 m; USCS symbol match; water level within 0.15 m; recovery and
  RQD when printed.
- Acceptance: ≥ 0.95 on N-values and sample depths on text-layer logs
  before WP3 starts; scanned logs scored separately once DI results exist.

### WP3: lab sheets → lab tests → DIGGS lab tests

- **`lab_reader`**: one sheet (or one multi-page test) per call; the test
  kind from the label and title; a typed result schema per kind; the sample
  link from the printed boring ID and depth; curves digitised only when no
  table gives the values (e–log p, gradation, stress–strain), using
  `render_region` on the plot area with the located axis ticks as the
  calibration. On scans, DI table cells are handed to the reader as cells.
- The lab **summary table** most reports carry ("Summary of Laboratory
  Tests") is read first and used by the reconciler as the cross-check for
  the per-sheet values.
- DIGGS: AtterbergLimitsTest, ParticleSizeTest (SieveAnalysis, Hydrometer),
  ConsolidationTest, TriaxialTest, DirectShearTest,
  UnconfinedCompressiveStrengthTest, LabCompactionTest, LabCBRTest,
  MoistureContent, LabDensityTest, SpecificGravityTest, LabChemicalTest
  (corrosivity), LabPermeabilityTest.
- Ground truth (mine): 30 tests across kinds and ≥ 4 labs (US firms' own
  labs, a local lab reporting in French, a 1990s typed sheet). Acceptance:
  ≥ 0.9 of index values exact; curve points within 2 % of the axis span.

### WP4: narrative + reconciler + the `report_ingest` sub-agent + library page

- **`narrative_reader`**: reads narrative segments with `read_document`
  (structure-aware, cited) and fills `general` and `natural_hazards` (§3).
  Scored on 8 reports against my hand answers, field by field, with the
  owner's earlier query outputs as a second reference where they exist.
- **Reconciler**: joins, unit normalisation, narrative-vs-appendix checks,
  the QA section; a value that differs between the summary table and the
  sheet is recorded as a conflict, not silently resolved.
- **Writers**: summary.md, page.md in the WikiLLM page format plus a SQLite
  index (`reports.db`) so many records can be searched the way the owner's
  library is; diggs.xml.
- **`report_ingest` graph**: LangGraph nodes 0–4 of §4; readers as
  sub-subagents with model-call budgets (`ModelCallBudgetMiddleware`) and
  `ScratchFilesystemGuard` like the existing sub-agents; attached as a
  `CompiledSubAgent` from `build_deep_agent`; one new primary tool; the
  same graph callable headless over a folder.
- Prompt: the primary delegates whole-report ingestion to `report_ingest`
  and never tries to read a 400-page report itself.
- Live check on the cluster: one public report (R36) end to end; the DIGGS
  file opens in `parse_diggs`; `plot_plan_view` and `plot_parameter_vs_depth`
  work from it; the summary cites pages that exist; one scanned report
  through DI.

### WP5: calc printouts and the long tail

- **`calc_reader`**: calc printouts (settlement, pile capacity, seismic,
  slope) → method, program, key inputs, key results, in prose with pages;
  not DIGGS.
- Scanned reports that DI did not cover: RapidOCR path measured on R13 and
  R38 after the cluster install check.
- Boring locations from the plan figure (the old repo's coordinate work);
  planlens IR plus `find_quantities` make this feasible.

### Parked

- Foundry/Spark packaging of the pipeline (the old repo's home platform).
- Word documents (nothing in the corpus needs it).

---

## 6. Gates, cost and risks

**Gates.** WP1 ≥ 0.90 on the five gated labels; WP2 ≥ 0.95 on N-values and
sample depths; WP3 ≥ 0.9 on index values; WP4 field-by-field score published
in the ledger before the sub-agent is wired in; every DIGGS file XSD-valid and
round-trip-equal; the existing app and planlens gates green.

**Cost shape** (order of magnitude; measured in WP2). A median report (~150
pages) has roughly 20 narrative pages, 30 log pages, 40 lab pages, 30 calc
pages. With one reader call per log page and per lab sheet plus a handful for
the narrative and reconciliation: about 100 model calls, roughly half with one
page image; DI on scanned pages only, cents per report. The count and the DI
page count are recorded per run in the QA section.

| Risk | Handling |
|---|---|
| Template variety (every firm's log differs) | No templates: headers and ruler are read off the page; measured across ≥ 6 templates before acceptance |
| Scanned and OCR-over-scan reports (9 of 38) | DI through planlens, RapidOCR fallback; `text_reliable` keeps garbage text out |
| Duplicate over-claim drops pages | WP0 fix; the ingest never skips a page on an image-only claim |
| Non-English lab sheets | Multilingual keyword lists; the readers are language-agnostic |
| Unit mix (ft/m, tsf/kPa, pcf/kN/m³) | Stored as printed with a unit string; converted once in the writer; DIGGS `uom` always set |
| Model invents a value | Typed schemas; provenance required; reconciler cross-checks against summary tables and narrative counts; QA lists partials |
| DIGGS that validates but is wrong | Round-trip through `parse_diggs` with value equality, not just XSD |
| Public repo | Corpus, DI results and measurements in gitignored `raw/`; IDs only in committed docs; the OBO-specific enumerations are field values, not project names |
| Cluster egress | No runtime downloads: RapidOCR models ship in-wheel; pydiggs schemas bundled; DI runs through the Funhouse SDK |

## 7. Open items for the owner

1. **TODO (owner): review the two query schemas in detail.** They were
   written for GPT-4o runs months ago and the outputs of those runs are gone,
   so there is no second reference; until the review lands the schemas are
   used exactly as given (owner's call, 2026-09-16: "run with what you
   have"). Any question added or changed is a field in `model.py` plus a line
   in the reader prompt.
2. Not blocking: DI into `raw/di/` for R38, the scanned pages of R03 and
   R33, and the unreliable-text pages of R02, R31 and R32 (58 pages in R32
   alone read as confident nonsense without it); a fresh export of R17's
   truncated DI result. The WP0 harness prints the exact page ranges.
3. Not blocking: the PDF behind the one unmatched hand-label sheet (153
   pages; its DI result is already in `raw/di/` as the unmatched file). Drop
   it into `raw/corpus/` as R39 and the scorecard returns to 15 reports.
4. **Repo visibility.** CLAUDE.md calls this repo private; the GitHub API
   reports it PUBLIC (checked 2026-09-16). Tracked field-feedback folders
   carry project names in their paths. Owner's call: make the repo private,
   or scrub. This train keeps everything private under gitignored `raw/`
   and IDs elsewhere regardless.
