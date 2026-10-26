# Report ingest — geotechnical report → organised record + summary + DIGGS 2.6

**Status: PARKED 2026-09-21 at 5.25.0 (released — tag `v5.25.0`, master
`46ae949`, on PyPI); pickup list in `HANDOFF.md` §0a-current.** WP0–WP7 are
BUILT and cluster scoring is still pending the owner's run: nothing in 5.25.0
has met the tier that will do the work. The next session's work is Tiny Apps
(`tinyapps/TINYAPPS.md`), not this train. WP0 and WP1b shipped in app
5.19.0 with planlens 0.5.0. WP2 through WP4 — the record, the log reader and
DIGGS 2.6, the lab reader, the narrative reader and the reconciler, the
writers, the deterministic graph, the folder runner and the app sub-agent —
plus the WP5 vision-first label experiment shipped as 5.20.0 with planlens
0.6.0. **WP5's calculation reader, WP6's pit and sounding readers and WP7's
report library are all BUILT and all carried by 5.25.0**, together with the
page-label VOTE (`label_vote.py`) and bound reports as their own records
(`bound.py`). What remains of WP5 is the long tail — the scanned reports DI did
not cover, and boring locations off the plan figure.

**Every model number for WP2–WP4 is still missing, and that is the state.**
What the ledger holds for the three readers is the baseline WITHOUT a model:
`log_grid` alone for the logs, the page's own detected tables alone for the
lab sheets, and nothing yet for the narrative. The development-engine
checkpoints were never run for any of the three. The numbers arrive from the
owner's four-stage `score_on_cluster` run, on the tier that will do the work
— which is why the sub-agent ships with `enable_report_ingest=False` and why
no reader's accuracy is quoted anywhere. The calc reader is in the same
state: its FLOOR is measured (no model) and the reader itself is not.

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
|       record.json -> summary.md, page.md (+ SQLite row), diggs.xml
|       (pydiggs XSD gate, parse_diggs round-trip gate)
|
+- 5. the library (added 2026-09-21)                     deterministic + 1 model
        report_library (CompiledSubAgent, one primary tool, OFF by default):
        Library(root) rebuilds its FTS5 index from the records whenever they
        are newer than it, and ten query functions answer ACROSS reports --
        list/filter, search, facts, compare, explorations, lab, calculations,
        disagreements, stats -- every row carrying the report id and the pages.
        The model is the APP's, the ceilings are Python, and every citation is
        checked against what the queries returned before the answer goes back.
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

### Two things the scoring run itself has to do (added 5.22.0)

- **The output is mirrored somewhere durable.** `/tmp` does not survive a
  cluster restart, and one wiped 38 finished label reviews (~$17 of calls).
  `report_ingest/mirror.py` copies every run file to SharePoint or to a
  durable folder as it is written, and restores a wiped `out_dir` at the
  start of the next call; a mirror failure warns and never stops the run.
- **Disagreement is a measurement, not a loss.** `stages=("vote",)` sets the
  rules, the vision pass and the review against each other with no model
  call: the agreement rate, a per-label trust table learned in sample only,
  three combining policies scored by the same scorer as the voters, and the
  accuracy a targeted review of the disagreements would need to carry the
  set over the gate (`report_ingest/vote.py`).

### The readers vote, and the pipeline runs on the cluster (added 5.23.0)

- **The floor is the first voter.** Built: `report_ingest/floor.py`,
  `log_floor.py`, `lab_floor.py`. The grid's rows (a log) and the page's
  detected tables and title (a sheet) are read into the record BEFORE any
  model call, at their own confidence; the model is shown that record and
  may add, may correct only with evidence, and may never drop. A split is a
  `disagreement` QA entry carrying both values and both confidences; a
  value the model omits is kept with a note. Motivated by cluster run 6–7:
  the log reader lost to the grid on seven of ten blind logs.
- **The `ingest` stage** runs `graph.ingest_report` end to end per report on
  the cluster — record, summary, library page, DIGGS with both gates,
  `qa.json` — reusing a saved label run's review, and scores the record
  against the hand truth with the same scorers. The whole-pipeline score.

### The page labels are a VOTE, and that is the label path now (added after 5.24.0)

**Step 0 of the architecture above -- "planlens: page labels (new)" -- is no
longer one voter.** The corpus run of 2026-09-20 (ledger runs 7 and 10) showed
planlens' rules and a cheap vision pass to be complementary by label class
rather than one being better (rules 0.908 in sample / 0.767 honest blind;
vision in sheet mode 0.655 / 0.867 at about $0.05 a report; the rules owning
`appended_report`, `other` and `calculation`, vision owning `plan`, `profile`,
`photos`, `cover`, `toc` and `figure` recall), and the label review breaking
nearly as many labels as it fixed on the reports its prompt was tuned against
while earning its $0.45 on reports it had never seen.

So the label path inside `graph.ingest_report` is now:

```
0.  planlens rules  ------0a. one vision pass ------ +--> label_vote.combine(policy) --> a label per page
    (cheap tier, sheet)    |     with its confidence, its voters
0b. the printed form -----/      and an `agreed` flag
    (log_templates, where a fingerprint file is in force)
                                            |
                                  the pages they SPLIT on
                                            |
0c. the label review, given ONLY those pages plus two either side,
    on a budget of max(20, 0.5 x split pages)
                                            |
1.  work items from the settled labels, as before
```

`label_policy` is `structural` by default and `rules` reproduces the old
behaviour exactly; `review_mode` is `disagreements` by default, with `all`
(every page, as before) and `none`. The arithmetic lives in ONE module,
`report_ingest/label_vote.py`, which the `vote` scoring stage calls too, so a
policy cannot mean one thing in the measurement and another in production.
`ReportRecord.page_labels` carries the chosen label, its confidence, every
voter and the `agreed` flag per page; a split the review does not settle is a
`QAEntry(kind="label_disagreement")`. This is item 1 of "Disagreement as a
signal" in `FUTURE_IDEAS.md`. **Measured 2026-10-04** (ledger
`module_work/field_feedback/2026-09-16_report-ingest/MEASUREMENTS.md`,
"FOUNDRY RUN 1"): on the in-sample reports the rules alone beat every policy
(0.896 vs structural 0.878, structural + review 0.863); on 60 never-opened
pages the vote beats the rules by two pages. Not enough to decide: the
default stays `structural` until more blind pages are hand-labelled (owner
item in `HANDOFF.md` §0a-current).

### A report bound inside a report is its own record (added after 5.24.0)

Triage's `bound_together` was a flag; it is now ACTED ON. Each bound page range
-- the union of what triage said and the record's own `appended_report` labels,
floor `MIN_BOUND_PAGES` = 4 -- goes round this same loop into its own record
under `out_dir/bound/<id>/`, with `ReportRecord.bound_documents` on the parent
and `ReportRecord.parent` on the child, so that an earlier firm's borings are
that firm's and the parent's counts are the parent's (`report_ingest/bound.py`;
`ingest_bound=False` restores the old listed-and-skipped behaviour).

### The calculations are read (added 2026-09-21)

A quarter of the hand-labelled pages of this corpus are calculation
printouts — 1,009 of them over eight of the fourteen labelled reports — and
until now every one was a QA entry saying the pages existed and had been
skipped. `calc_reader.read_calculation` is the FOURTH reader and takes one
run of `calculation` pages (the graph already groups them, splitting a
calculation appendix on program banners and printed titles), returning ONE
`Calculation`: what it works out from a controlled list of thirteen kinds,
the program and version as printed (`null` where none is named, which is the
right answer for a spreadsheet), the method, what it is for, the labelled
values it was GIVEN and the labelled values it WORKED OUT, and a summary of
sixty words.

It has a floor like the other two — every (label, value, unit) a pattern can
read off the pages' tables, their ruled data tables' last filled row, their
label-and-value spans on one printed line, and planlens' own `quantities`
pass — and the same merge rule: add, correct only with evidence, never drop.
Its gates are three: a RESULT whose number is on none of the pages to the
precision it was reported at drops to confidence 0.3 and is listed; a kind
outside the list becomes `other` with a note; a unit the record cannot
convert keeps the value as printed and says so on the unsettled list.

**DIGGS ignores calculations, deliberately.** DIGGS 2.6 is an interchange
format for what was OBSERVED in the ground: it has no element for a method
and no home for a chosen thickness. The record, the summary page's
"Calculations" section and the library page carry them instead, each value
with the page it was printed on.

### The pits and the soundings are read (added 2026-09-21)

`InvestigationKind` has said `test_pit`, `cpt` and `dcp` since the first
version of the record, and until now every one of them was read as though it
were a boring. A pit lost the hole it actually was; a sounding lost the whole
of what it measured, because a depth SERIES of three channels has nowhere to
go among layers, samples and driven records. The corpus carries **251 test pit
pages, 54 DCP pages and 23 CPT pages** in nine of the fourteen labelled
reports.

**A test pit is a LOG.** A pit form has a ruler, a description column and
samples, so the log reader takes it and the log's floor seeds it. What is new
is `Investigation.pit`: a length, a width and the pit's own depth. **No pit in
this corpus prints a labelled plan size** -- fourteen were checked across four
reports -- but every one names its BUCKET in the equipment field, and a trench
dug with a 90 cm bucket is 90 cm wide. That is read as the WIDTH at a lower
confidence, with the provenance saying where it came from. A pit photographed
with a sketch (no ruler) goes to the model from the picture with a floor of
header fields only, and is the one log whose depths are accepted without a
fitted scale.

**A sounding is NOT a log**, and `sounding_reader.read_sounding` is the FIFTH
reader. Two shapes of sheet, read differently, and the scorecard prints the
split:

* **TABULATED** -- the sheet's own table is the record, read deterministically
  off planlens' detected tables or, where a sheet rules no table at all, off
  the text lines clustered into columns. One call then goes on the header, and
  the brief tells the model NOT to re-type the rows.
* **PLOTTED** -- the floor is the header and the AXIS RANGES read off the
  plot's own text, and the traces are digitised through `zoom_plot` at a fixed
  step, a point per depth, with a confidence that falls where one trace crosses
  another. The axis ranges are what makes a digitised reading checkable.

**DIGGS 2.6 has a home for a cone sounding and none for a dynamic cone.**
`StaticConePenetrationTest` is the schema's own procedure for the first, whose
dictionary entry names `tip_resistance`, `sleeve_friction` and
`pore_pressure_u2` as the properties that occur under it. There is no
`DynamicConePenetrometerTest` anywhere in the published schema, so a dynamic
probe is written as the `DynamicProbeTest` the schema DOES declare -- "all
methods that involve driving a rod by impact hammer". A sounding is ONE Test
with a ResultSet of many ROWS and the depth of each reading is a COLUMN of the
set; `parse_diggs26_soundings` reads it back row by row.

**And the calculation grouping defect is fixed.** The calc reader's own hand
truth found the item grouping folding four different calculations into one
fifteen-page item where an appendix runs printouts back to back;
`split_calc_items` splits a run where a page's title, set as a title, names a
different calculation, where the program changes, or where a banner returns;
a run stays one item up to `MAX_RUN_PAGES` (22, what the reader's two calls
show) and a longer one is cut into balanced parts (RI-17; see the
report_ingest README).

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
- **Built, 5.23.0 — the floor.** `log_floor.seed_from_grid` seeds the record
  from the grid before the call; `merge_investigations` folds the model's
  answer onto it (add / correct with evidence / never drop); one follow-up
  call on the reader's own unsettled list with the rows magnified. The
  scorecard prints the grid, the model alone and floor+model.

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
- **Built, 5.23.0 — the floor.** `lab_floor.floor_from_tables` seeds typed
  tests from the title's kind, the printed link and the detected tables
  (labelled values, grading series, summary rows); `merge_lab_tests` folds
  the model's answer onto them under the same rule. No sheet can now score
  below its tables.

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

- **`calc_reader` — BUILT, 2026-09-21; carried by 5.25.0.** Calc
  printouts (settlement, pile capacity, seismic, slope, pavement, retaining
  walls …) → one `Calculation` per run: kind from a controlled list, program
  and version as printed, method, subject, labelled inputs and results with
  the page and box of each, a sixty-word summary, and what it could not
  settle. A deterministic floor, one model call (two at most), three Python
  gates, `zoom_plot` for a result printed on a drawing. Wired into the graph
  (`ITEM_READERS["calculation"] = "calc"`), the writers (a "Calculations"
  section in the summary and the library page) and the reconciler (a subject
  naming a boring links to it; a result that contradicts the narrative's
  bearing pressure, settlement or site class is a `disagreement`). **Not
  DIGGS**, and the DIGGS writer's docstring says why.
  - Scored by `report_ingest/calc_scoring.py`, cluster stage `"calc"`, local
    twin `module_work/report_ingest_harness/measure_wp5_calc.py`.
  - **Hand truth: ten runs, six reports, seven kinds, 44 pages**, in the
    private `raw/truth/calc/` with its own README and protocol.
  - **The floor alone, 2026-09-21**: program 80 % (8/10), inputs 73 %
    (64/88), results 48 % (36/75). The reader itself is unmeasured until the
    cluster run.
  - **OPEN ONLY — there is no blind calculation truth set**, and every
    number the stage prints is in sample until there is one.
- Scanned reports that DI did not cover: RapidOCR path measured on R13 and
  R38 after the cluster install check.
- Boring locations from the plan figure (the old repo's coordinate work);
  planlens IR plus `find_quantities` make this feasible.

### WP6: test pits, cone soundings and dynamic probes (2026-09-21) — BUILT, carried by 5.25.0

- **`sounding_reader.py`** reads a `cpt_log` or `dcp_log` item into an
  `Investigation` with a `CPTData` or a `DCPData`; the log reader gained the
  pit's dimensions and the no-ruler photograph path.
- **Floor:** the sheet's own table where it tabulates, the axis ranges where it
  plots. Merged under the package's one rule -- add, correct only with
  evidence, never drop.
- **Gates:** a depth outside the sheet's own depth axis, a channel outside its
  printed range by more than a tick, a negative tip resistance and a series
  with no depth unit are refused with the range they fell outside.
- **DIGGS:** `StaticConePenetrationTest` and `DynamicProbeTest`, through both
  gates, read back at each row's own depth.
- **Scoring:** `sounding_scoring.py` (a pit by the log scorer's metrics plus
  `dimensions`; a sounding per depth step, within max(5 %, one axis tick) for
  qc, fs and the index, 10 % for u2, exactly for a blow count), cluster stage
  `"soundings"`, the local twin `measure_wp6_soundings.py`, and the `ingest`
  stage's record scoring.
- **Hand truth:** thirteen sheets over six reports, all OPEN.
- **The floor alone, 2026-09-21:** 73 % overall (257/353), which is 96 % on the
  four tabulated sheets and 31 % on the nine plotted ones. **The readers
  themselves are unmeasured** until the cluster run.
- **OPEN ONLY -- there is no blind sounding truth set**, and every number the
  stage prints is in sample until there is one.

### WP7: the report library (2026-09-21) — BUILT, carried by 5.25.0

- **`library.py`.** `Library(root)` over the folder the ingest writes. The
  index is DERIVED and the records are the truth: a folder restored from
  SharePoint with no `reports.db`, or one whose records were rewritten since,
  rebuilds on the next question — the writers' own `reports` rows AND the
  full-text index — checked by mtimes against a stamp in `meta` and keyed off
  each `report.page.md`'s front matter. Nothing is written back into a record.
- **Ten queries**, every row carrying the report id and the PDF pages:
  `list_reports`, `find`, `where_is`, `facts` (the 37 narrative fields, the
  unanswered ones NAMED), `compare`, `explorations`, `lab_summary`,
  `calculations`, `disagreements` (the six QA kinds a person should look at),
  `library_stats`. A bound report is a report of the library in its own right.
- **Search is FTS5 the way the reference layer does it** (contentless external
  content, `porter unicode61`, BM25 with the subject weighted), over chunks
  from the RECORD and from the written page and summary. The **rapidfuzz
  fallback runs on the field VALUES**, not on whole chunks, and only when the
  full-text query comes back thin.
- **`library_agent.py`.** `report_library`, a `CompiledSubAgent` plus one
  primary tool, **OFF by default** and feature-detected on the FOLDER rather
  than on a package version, bound to the APP's own chat model. Its prompt's
  one rule is that every fact comes from a tool result in that conversation,
  cited `(report id, page)`, with a `Gap:` line per thing unsettled. The
  ceilings are Python (eight queries an answer, 25 rows and 4,000 characters a
  result) because deepagents reads no middleware on a compiled sub-agent, and
  the **citations are CHECKED, not copied**.
- **Measured 2026-09-21, no model and no network** (`measure_wp7_library.py`,
  twenty questions over a synthetic library of six records): the chosen query
  scores reports **0.949 / 1.000** and pages **0.939 / 1.000**; search alone
  scores reports **0.647 / 0.943** and pages **0.417 / 0.323**.
- **The model half is UNMEASURED.** The cluster cell is in
  `report_ingest/README.md` under "The report library"; it reads a private
  `library_questions.json` and is scored on the report ids the answer CITES.
  `report_ingest/library_questions.EXAMPLE.json` ships in the wheel and shows
  the shape.

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

   **Where the rulings go (5.24.0): `report_ingest/narrative_glossary.py`.**
   The eight-report cluster run showed most of the narrative reader's misses
   were CONVENTION rather than comprehension — null where the hand writes 0,
   the wrong sense of a "mention" question, a hazard list with seismic shaking
   in it — so the house rules are now data in that one file, each with a
   STATUS. Six rules are in force and marked **DRAFT** pending this review:
   counts of explorations the report has none of are `0` not null; `null`
   means the report does not say; the four "mention" fields answer on whether
   the report DISCUSSES the topic anywhere; `postName` is the city of the
   diplomatic post; `primeAe` is the architect-engineer of record; and
   `earthHazardsExposed` uses the listed phrases only and never includes
   seismic shaking. Confirming one is changing its `status` to `CONFIRMED`;
   overruling one is editing the sentence. Nothing else changes — the block
   goes into the reader's prompt and the reader answers under it. Field names
   stay verbatim, and `unknown_fields()` plus its test forbid a rule keyed to
   a name the schemas do not carry.

2. Not blocking: the PDF behind the one unmatched hand-label sheet (153
   pages; its DI result is already in `raw/di/` as the unmatched file). Drop
   it into `raw/corpus/` as R39 and the scorecard returns to 15 reports.

3. **The INGEST LEDGER is the owner's design and is NOT built** (2026-09-21).
   One function the ingest calls at the END of every report, whether it ran in
   the app, on the cluster or from `run_folder`, writing ONE append-only row
   about that report to every sink it can reach: a JSON-lines file in the
   workspace folder mirrored to SharePoint (that pair is the source of truth),
   a Delta table REBUILT from that file rather than written alongside it, and a
   SharePoint list as the human view where the client supports one — the
   optional sink, whose absence must never fail an ingest. The row carries the
   file hash and name, pages, ingested-at, package version, the deployments
   that actually served the calls, document type and workflow, counts by kind,
   narrative fields answered and null, QA counts by kind, DIGGS written and
   validated, tokens and dollars, where the outputs live, and the run id. **A
   re-ingest appends a new row and never updates one.** Written up in full in
   `module_work/FUTURE_IDEAS.md` under "The ingest ledger"; nothing of it
   exists in code.
4. **BLIND truth sets are OWED for the calculations and for the soundings.**
   Both readers shipped with hand truth that is entirely IN SAMPLE — ten calc
   runs over six reports, thirteen pit and sounding sheets over six reports,
   every one of them read while the prompt was being written. `OPEN.txt` in
   each folder names all of them and the scorecard prints that there is no
   blind set where the open/blind line would go, so no number either stage
   reports is evidence about a report nobody has looked at. Two sets of roughly
   the same size, truthed from reports neither prompt has seen, is what closes
   it.
5. **The narrative reader's cheap-vision SECOND VOTER is designed and not
   built.** The page labels have three voters and every extracted value carries
   a method and a confidence; the narrative answers have ONE voter and no
   second opinion. The design is the same shape as the label vote: one cheap
   sheet-mode pass over the pages the narrative reader cited, answering the
   same fields, with agreement raising confidence and a split becoming a
   `QAEntry` a reviewer is sent to. It waits on the cluster numbers for the
   5.24.0 levers, because tuning a second voter against an unmeasured first one
   is measuring noise.
6. **DI is owed for the scanned reports, and only the owner can run it.** Not
   blocking: DI into `raw/di/` for R38, the scanned pages of R03 and R33, and
   the unreliable-text pages of R02, R31 and R32 (58 pages in R32 alone read as
   confident nonsense without it); a fresh export of R17's truncated DI result.
   The WP0 harness prints the exact page ranges. Without them those pages are
   read from their own text layer alone, and where that layer is wrong the
   reader is reading nonsense confidently.
7. **`pydiggs` is NOT on the cluster**, deliberately: it is an optional extra
   (`[pydiggs]`), demoted in 5.13.0 because it dragged Sphinx and a Pygments
   upgrade into the deployment. So a DIGGS file a cluster run writes is checked
   by the ROUND TRIP — read back through the app's own parser and compared
   value by value — and the XSD line in `RESULTS.md` reads `not checked here`
   rather than `valid`. The schema gate runs in the offline gate, where pydiggs
   is installed. Read a cluster DIGGS row accordingly.
8. **Repo visibility.** CLAUDE.md calls this repo private; the GitHub API
   reports it PUBLIC (checked 2026-09-16). Tracked field-feedback folders
   carry project names in their paths. Owner's call: make the repo private,
   or scrub. This train keeps everything private under gitignored `raw/`
   and IDs elsewhere regardless.

## 8. MODEL-FIRST INGEST (RI-87, owner 2026-10-25)

**Status: design, not built.** The owner's decision (OWN-96, 2026-10-25):
"Yes, turn report ingest model first." This section is the design. Its build
pieces (§8.9) go on BACKLOG RI-87 and nowhere else. Today's pipeline stays
the default until the measurement in §8.8 says otherwise.

**The rule this design follows** (owner, 2026-10-25): "Simple is better, I
think we got too complex chasing narrow goals." So the plain path is built
and measured first, and code is added to it only where it measurably beats
the plain path. Everything below that is marked as a fallback is built only
if its trigger is met.

### 8.1 What the evidence says

- **A plain call reads about as well as the pipeline.** RI-86 handed Opus
  5.5 each hand-truthed item's own pages as one PDF, one short instruction
  and the stage's answer schema as the answer tool: one call, no brief, no
  floor, no merge. Blind logs 175 of 214 against the pipeline's 182; blind
  lab 420 of 459 against 417; calc 176 of 203 against 181 (open set);
  narrative R05 recall 24 of 27 against 25. The call cost about half as much
  (MEASUREMENTS 'CLAUDE LOCAL RUN 2').
- **Where the pipeline still leads**, the floor and merge fill fields the
  answer schema has no slot for (RI-82: recovery printed as a length, a
  summary row's bottom elevation, the project name and number on a log), and
  the pipeline gives a box on the page (logs 557 of 619 values, lab 71 of 72
  tests, every calc value; the plain answer places none).
- **The floor adds values only on logs.** The pipeline's record against its
  own model's answer alone: logs +9 open (287 to 296) and +15 blind (167 to
  182); lab +0 on both sets (351, 417); calc -1 (182 to 181).
- **The plain arm exposed two pipeline defects**: the log depth gate refused
  six water depths on R15_p46 (RI-76), and a scanned compaction sheet was
  read as one summary table (RI-81, R15_p184).
- **Page labels**: RI-73's range labeller (Haiku describes every page, Opus
  labels by sections over the descriptions, conventions B) sends 50-56 of
  3,418 pages to the wrong reader, against 290 for master's labels. In
  sample; on branch wip/ri73 behind `GEOTECH_INGEST_RANGE_LABELS`.
- **Models** (RI-78): lab on Opus 5.5; logs and calc on Sonnet
  5.5. That was measured on the pipeline. The plain call was measured on
  Opus only. The narrative on Opus 5.5 with every question in one call (owner
  2026-10-25, from their own runs: Opus with all the questions, or Sonnet 5.5
  only as one call per category of questions; Opus is likely cheaper in the end).

The plain arm was handed the right pages from the answer sheets. Model-first
has to find them itself, so the whole-report wave (§8.8) is the one that
decides.

### 8.2 The flow

The switch is **`GEOTECH_INGEST_MODEL_FIRST`**: `off` (the default; unset or
`0`) is today's pipeline, byte-identical (§8.2.2); `on` (or `1`) makes the
model's answer the record. `ingest_report(..., model_first=None)` follows the
variable; a string names the setting directly. A second variable says how
pages reach the model: **`GEOTECH_INGEST_MODEL_FIRST_PAGES`** = `auto` (the
default: a PDF where the engine can send one, otherwise pictures), `pdf` or
`pictures` (§8.7).

**Under `on`, only the model reads values into the record.** The floors
(`seed_from_grid`, `floor_from_tables`, `floor_from_pages`) and the merges do
not run. A value the code reads by itself (today, only the narrative's
caption counts) goes to the QA list as "read by the code, not by the model"
and never into the record. The schema additions of §8.5 are what close the
gaps the floor used to fill (recovery as a length, a row's bottom elevation,
the project fields).

Under `on` one report runs as below. "Code" marks a step with no model call.

1. **Open** (code). planlens opens the PDF and puts any Azure DI text into
   the same frame, as today. `facts_from_document`, `assign`,
   `document_outline` and `document_facts` run as today: the rules' labels
   are one voter and the outline goes to the labeller.
2. **Triage** (one call, as today). It still decides the workflow
   (`needs_person` stops; `appendix_only` skips the narrative) and the
   ranges of bound documents.
3. **Describe every page** (Haiku 5.5, one call a page; RI-73's
   `describe_pages`). Each description is saved under
   `out_dir/descriptions/` as it is made, so a run resumes. On a scanned page
   the describer reads the picture.
4. **Label by sections** (Opus 5.5, one call per report or per 300-page
   window; RI-73's `label_by_ranges` with conventions B).
   `label_vote.combine` takes the range label for every page it labelled; a
   page it left unresolved keeps the rules' label with a
   `label_disagreement` entry, as RI-73 already does. **The range labels
   replace the vision voter and the label review** (decided, §8.10): neither
   runs under `on`.
5. **Bound documents** (one identity call each, as today, `bound.py`). Each
   child goes round this same flow over its own pages. Its descriptions are
   already on disk; it gets its own range-label call.
6. **Group the pages for each call** (code; §8.2.1): one log per call, lab
   sheets in windows, one calculation printout per call, the narrative in one
   call or a few.
7. **Read** (one call per group; lab and narrative on Opus, the rest on Sonnet). The call
   carries the pages (one PDF, or each page as a picture with its text), the
   stage's short instruction (RI-86's, moved into the package) and the
   stage's answer schema with the additions in §8.5, offered as the `answer`
   tool through `ChatModelEngine`. No brief, no grid rows, no floor, no zoom
   tool.
8. **Build** (code). The stage's own `_Builder` turns the answer into record
   objects. Its checks run in the flag-not-drop form of §8.6.
9. **Place** (code; §8.4). Each value gets its page checked and a box where
   the page's printed text can give one.
10. **Reconcile, write** (code, unchanged). `reconcile`, the checklist,
    `write_outputs`, the DIGGS writer and the library run on the record as
    today.

Cone soundings and dynamic probes keep today's sounding reader under `on`. A
sounding is a series that code digitises, and no plain call was measured on
one.

#### 8.2.1 Which pages go in each call

The groups come from the range labels and the page descriptions alone. All of
it is code, and it is scored offline against the answer sheets' item pages
before any call is paid for (piece 3). Today's other grouping signals
(planlens' item key, the lab floor's specimen starts) are added back only if
that score shows the descriptions miss something they catch.

- **Logs: one log per call, all its sheets together.** A new log starts at a
  page whose description prints an exploration identifier other than the
  current log's, unless its `continues` field says it carries on the current
  one ("sheet 2 of 3"). A page that prints no identifier joins the log before
  it. This also fixes a scan problem of today's path: with no text layer
  every scanned log page has the same planlens key, so twenty scanned logs in
  a row become one item. A log longer than 12 pages (rare) is cut at a sheet
  boundary into two calls; the second part's layers, samples, drives and
  water readings are appended to the first's, the header is the first
  part's, and a QA note names the cut.
- **Lab: windows of at most 12 pages** (`MAX_LAB_PAGES`), cut by
  `lab_windows`, with the cuts moved to pages where the description's title
  or specimen identifiers change. Each test's `pages` are the pages it cites
  inside the window, not the whole window, so the whole-record scorer never
  sets one sheet's test against another sheet's answer sheet.
- **Calc: today's grouping.** `split_calc_items` cuts runs by each page's
  head. A printout of 12 pages or fewer is one call. A longer one (up to 22
  pages) is two calls through `page_window` (the first and last page in
  both), folded by `_fold_readings`.
- **Narrative: the narrative pages and the front of the report**, as
  `reading_pages` picks them today (cover, letter, contents and the first
  pages), plus the figure and table pages of the main body. Up to 100 pages
  inside the narrative's token budget go in one call. A longer narrative is
  read in consecutive parts and merged by the narrative builder's own rule
  (the first answer kept, lists joined, every disagreement between parts
  recorded), with one summary call after (`_summarise`), as today.
- **Limits.** The Anthropic API takes at most 100 pages and 32 MB in one PDF;
  a cut larger than 8 MB is sanitised (`cut_pdf`, from the plain baseline). A
  reply that stops at its output cap gives no answer (`Reply.cut_off`). That
  group is asked once more as two halves, split at a sheet boundary, and a
  half that is still cut off becomes a `skipped` entry naming its pages.

#### 8.2.2 Off is today

Every new piece of code runs only under `on`. The shared functions it reuses
gain parameters whose defaults are today's behaviour (the builders' check
mode). No field is added to the record (§8.5 says where the new values go),
so `report.record.json` keeps its keys; the setting a run used is written in
`run.json`, and in one QA note (`where="model_first"`) only under `on`. A
test runs the scripted-engine graph with the switch unset and checks that
every request and every output file is byte-identical to a run of master; a
second checks that no model-first module is imported.

### 8.3 What is kept, what is reused, what is bypassed

| Module | Kept as is | Reused differently under `on` | Not run under `on` |
|---|---|---|---|
| `graph.py` | triage, bound documents, the reconciler call, writing, the cost tally, per-item resume | `_read_items` takes the model-first readers' results, which keep today's dict shape (`investigation`, `tests`, `calculation`, `general`, `unresolved`, `cost`; `disagreements` and `kept` empty), cached as `items/mf-<key>.json` so a resume never mixes the two | `_vision_labels`, `_review` |
| `label_vote.py` | `combine` (with RI-73's range voter), `split_pages`, the policies | the range label decides each page it labelled | the vision voter's input; `neighbourhood` (no review) |
| `range_labels.py` (wip/ri73) | `describe_pages`, `label_by_ranges`, conventions B | the descriptions also drive the grouping (§8.2.1) | |
| `log_reader.py` | `LogReading` and its parts, `_Builder`, `_depth_window` (from `log_grid_for` with visual scales off: planlens' grid, no call) | the builder's checks in flag mode (§8.6); the grid's ruler is also a placement anchor (§8.4) | `read_log`: the brief, the grid rows, the zooms, the follow-up calls |
| `log_floor.py` | | | `seed_from_grid`, `merge_investigations` |
| `lab_reader.py` | `LabSheetReading`, `_Builder`, `lab_windows`, the kind vocabulary | the checks in flag mode (§8.6); each test's pages narrowed | `read_lab_sheet`: the brief, `zoom_plot`, the per-page floor; `_specimen_starts` |
| `lab_floor.py` | | | `floor_from_tables`, `merge_lab_tests` |
| `calc_reader.py` | `CalcReading`, `_Builder` (given no pool of printed numbers, as the baseline did), `split_calc_items`, `page_window`, `_fold_readings` | `_precision_tol` and `_on_the_page` become placement's number match for every stage (§8.4) | `read_calculation`: the brief, the page pictures; `floor_from_pages`, `merge_calculation` |
| `narrative_reader.py` | `NarrativeReading`, `_Builder`, `check_quotes`, `_summarise`, `reading_pages`, `_vocabulary_block` (the glossary) | the builder gets no caption count and no logs, as the baseline did; `count_captions` runs (text only, no call) and a count that differs from the model's answer is a QA note, not the record's value | `read_narrative`: the text chunks, the brief, the retrieval passages, the pictures of text-less pages |
| `sounding_reader.py` | all | | |
| `floor.py` | | | all (the fallback of §8.4 would use `settle`) |
| `engine.py` | `ChatModelEngine`, `strict_schema`, the answer tool, RI-80's unwrap | gains a PDF block (§8.7) | |
| `model.py`, `reconciler.py`, `checklist.py`, `writers.py`, `diggs_writer.py`, `library.py`, the scorers | all | | |
| `subagent.py`, `run_folder.py` | all: they call `ingest_report` and follow the variable | | |
| `run_plain_baseline.py` | its scoring | `INSTRUCTIONS`, `instruction`, `cut_pdf` and `PdfChatEngine` move into the package and the script imports them | |

### 8.4 Provenance after the answer: the text match

The model gives the page (the report's page index: the instruction maps each
PDF page to it, as RI-86's does) and no box. The answer schema's `bbox` is
ignored under `on`, because the model was shown no rows or cells to copy one
from. One new module, `report_ingest/locate.py`, finds the box after the
answer:

1. **Check the page.** Decided once per answer, not value by value
   (`report_ingest.checks.pdf_local`): when every page the answer cites is a
   PDF-local index (0 to n-1) and at least one is not a page of the call,
   all its cites are mapped to the report's pages, each with a note: the
   class RI-74 (6) fixed for one-page printouts, applied to every stage.
   Otherwise every cite is the report's page. When the call's pages start
   below n and the answer cites both pages of the call and pages outside it,
   the answer gets one QA note saying the numbers are ambiguous and which
   reading was taken. A page outside the call on that reading keeps the
   value, takes the call's first page at confidence 0.4 and a note, as the
   builders do today.
2. **Find the object's printed words on its cited page**, among the words
   that carry boxes: `Document.page(p, words=True)` gives the text layer, or
   the DI words on a scan, in planlens' one frame. An object is placed by the
   first of its anchors that is found:
   - a log layer by its description's words, a sample or drive by its label,
     then by its printed depth on the row the ruler puts at that depth
     (`Ruler.y_at`; the grid already runs for the depth check, §8.6), a water
     reading by its printed depth;
   - a lab test by its sample label, then its test title;
   - a calc value by its label and its number on one printed line;
   - a narrative answer by its quote, which is `check_quotes` as today. No box
     field is added to `Citation`; a viewer that wants one finds the quote
     with planlens' search.

   Text is matched with `Document.search(..., fuzzy=True)`, a number at its
   printed precision (`_precision_tol`, `_on_the_page`). One candidate gives
   its box, method `model` and the note "box from the text layer" (or "from DI
   words"). Several with no anchor between them give the page only and a note
   with the count. A log value need not be printed itself (a layer boundary
   is often drawn, not printed): its object's box is enough.
3. **The call's other pages.** An object found on one other page of the call,
   and only one, moves to that page with a note.
4. **Not placed.** The object stays in the record with its page and no box.
   On a page with text, its confidence drops to 0.6 and the item's QA note
   counts it ("N of M not found printed on the cited page"). This is today's
   calc result check, applied to every stage, and it never removes a value.
   On a page with no text layer and no DI text nothing can be placed and
   nothing is lowered; the note says the page has no text.

**Measured first, offline.** Piece 5 runs the text match on the saved RI-86
answers (Opus, the plain arm, today's schema) at no cost: objects placed, by
stage, on pages with text, beside the pipeline's 557 of 619 log values and
71 of 72 lab tests.

**The fallback, built only if a stage misses the gate.** If the text match
places fewer than 90 % of values on pages with text in a stage, that stage
gets the floor as a source of boxes: its floor runs, the stage's merge pairs
the floor's objects with the model's by the rules it already has, and a pair
gives the model's object the floor's box. The model's value always keeps the
slot, and no floor value enters the record. It is a `place` policy on
`settle` and on that stage's `merge_*` (default: today's), with tests that the
default reproduces today's merges.

### 8.5 Schema additions (model-facing)

The answer schema's text is what the model reads, so each addition is listed
with the description it carries. They go into model-first subclasses in
`report_ingest/model_first.py`. `LogReading`, `ReadSample` and `ReadRow`
themselves do not change, so with the switch off the answer tool's schema is
today's. Nothing else in the schemas changes: the rest stays as the plain
baseline measured it. These are what close the floor's lead (§8.1) without
the floor.

`LogReading` (logs):

| Field | Type | Description the model sees |
|---|---|---|
| `project` | str, default "" | "the project or job name the log prints in its title block, as printed; empty if none" |
| `project_number` | str, default "" | "the project, job or contract number the log prints, as printed; empty if none" |
| `client` | str, default "" | "the client or owner the log names, as printed; empty if none" |
| `header_fields` | list of {`name`, `value`}, default [] | "every other header field the log prints that has no field here: its label and its value, as printed" |

`ReadSample` (logs; `LogReading.samples` retyped to the subclass):

| Field | Type | Description the model sees |
|---|---|---|
| `recovery_length` | float or null | "recovery printed as a LENGTH, a number under a heading such as REC (in) or RECOVERY (cm); null when it is printed as a percent or not at all" |
| `recovery_length_unit` | str, default "" | "the unit of recovery_length as the column heads it: in, cm, ft or m" |
| `recovery_percent` | (exists) | description gains: "only when the log prints a percent" |

`ReadRow` (a lab summary table's row; `ReadTest.rows` retyped):

| Field | Type | Description the model sees |
|---|---|---|
| `elevation_bottom` | `ReadQuantity` or null | "the bottom of the specimen's elevation range, when the table prints a range; null otherwise" |

Where the builder puts them, with no record field added:

- `project`, `project_number` and `client` go into `Investigation.fields`
  under the keys the log scorer's `FIELD_MAP` reads (`project`,
  `project_number`, `client`: the grid's names). `header_fields` go in under
  the label as printed, or under the grid's name where planlens' header
  vocabulary has one.
- `recovery_length` goes into `Sample.recovery` (a `Quantity`), which the
  record already has and the scorer already reads (`_recovery_numbers`).
- `elevation_bottom` goes where the scorer looks for it. Today the lab floor
  reads an elevation column into the row's depth slots (`lab_floor`'s column
  vocabulary), so piece 4's builder checks R28_p159's answer sheet offline
  and uses `SummaryRow.other` if the row has no field for it.

`ReportRecord.project` is still filled from the narrative (`_project_from`),
as today.

### 8.6 The checks that stay

Every check today's builders run stays. Under `on` they flag rather than drop,
and they split into two kinds:

- **Impossible** (no real sheet prints it: a fraction outside 0 to 100, a
  water content or an Atterberg limit below 0, a liquid limit below the
  plastic limit, a grading in which more passes the smaller sieve, a depth
  past 300 m, a negative depth). The value stays out of
  the typed slot, so nothing downstream (DIGGS, the reconciler, `si_numbers`)
  takes it as a reading, and a QA entry of rank `open` carries the values as
  read. The record's own validators (`ge=0, le=100` on a percent) would
  refuse it in the slot anyway.
- **Implausible** (a check on where it sits on the paper, or which number it
  is). The value stays in its slot at a lowered confidence, with an
  `out_of_range` or `note` entry.

| Check | Today | Under `on` |
|---|---|---|
| log depth window (`_Builder._depth`) | outside the ruler's window: refused, and a layer or sample whose top is refused is dropped whole; no ruler on a boring: every depth refused | **implausible.** Kept at confidence 0.5, one `out_of_range` entry per value naming the window. No ruler on any page: every depth kept, and one note per log ("no depth scale found; the depths are as read, not checked against a ruler"). The window applies only to values placed against the ruler (layers, samples, drives). A water, casing or cave depth is checked against 0 to the total depth plus the window's slack; with no total depth it is kept and counted in the log's one note. This is the class of RI-76 (5) on R15_p46: six water depths printed in a table on the sheet, refused for lying outside the ruler's window. It is also most of RI-79's misses on scans. |
| log total depth negative | refused | impossible: as now |
| lab depth outside 0 to 300 m | refused | impossible: out of the slot, `open` entry with the value |
| lab percent outside 0 to 100 | refused | a fraction (gravel, sand, silt, clay, fines, percent passing, saturation, organic, ash) outside 0 to 100: impossible, the same. A water content (test, specimen, optimum, compaction point) or an Atterberg limit (LL, PL, PI, shrinkage limit): impossible only below 0, kept above 100. The water content can be greater than 100 % (FHWA-NHI-06-088 Vol. I §2.1.2, p. 2-5); DM 7.1 (UFS 3-220-10) §8-2.2.2 cites clays with liquid limits to 143 and PIs to 112; GEC-13 §3.7.4 (p. 3-53) has water contents of 94 to 364 percent. |
| log recovery percent over 100 | clamped to 100 | out of the slot (the record's field holds 0 to 100), `open` entry with the value and words that it is a reading, not a misreading: recoveries over 100 % are recorded (FHWA-NHI-06-088 p. 3-44). Below 0: impossible |
| log drive lengths that do not line up with the blows, or have no unit | dropped without a word | out of the drive, `open` entry with the lengths, the blows and the unit as read |
| log word outside a vocabulary (sample kind, exploration kind, water `when`) | `other` / `unknown`, without a word | `other` / `unknown`, with an entry naming the word as read |
| log pit dimension with no unit | refused | kept in no unit, one note, as the total depth is |
| lab liquid limit below plastic limit | all three refused | impossible: all three out of their slots, one `open` entry with the three values |
| lab grading that rises | the series refused | impossible: the series out, one `open` entry |
| lab kind outside the vocabulary | `other` with a note | as now |
| calc value printed on the pages (`_on_the_page`) | confidence lowered | moves into placement (§8.4 step 4), for every stage |
| narrative quote check (`check_quotes`) | confidence lowered, listed | as now |
| narrative caption counts (`count_captions`) | the code's count is stored over the model's answer | the model's answer is stored; a code count that differs is a QA note "read by the code, not by the model" |
| narrative `structureCount` from `structureList` (OWN-08) | the list's length is stored | as now: it is the owner's rule over the model's own answer |
| narrative builder refusals (outside an enumeration, a citation off the narrative's pages, a bearing value with no unit, past a word limit, two parts disagreeing) | refused or recorded | as now |
| a page outside the item | value kept, page and box dropped | §8.4 step 1 |

The flag mode is a builder parameter (`checks="refuse"` today, `"flag"` under
`on`). RI-76's fix on today's path stays its own item, because it changes the
default path and is measured on its own.

### 8.7 Calling the model on Foundry

The run package (INF-40) is the AI FDE running a lightweight Python transform
that calls `score_on_cluster` with a chat model, read through
`ChatModelEngine`. Claude reaches that code by one of two routes. What the
repo shows about a PDF document block on each:

- **The LLM proxy** (Claude's default route on Foundry;
  `webapp.engine_config.foundry_chat_anthropic_class`, a `ChatAnthropic`
  against `/language-model-service/api/proxy/anthropic`). It takes the
  Anthropic Messages shape, and `langchain_anthropic` turns LangChain's
  `file` block into an Anthropic `document` block, which is how the plain
  baseline sent its PDF over the direct API. The AI FDE's measurements of the
  proxy (2026-10-20) cover cache marks, `service_tier`, thinking and
  `tool_choice`; no document block has been sent through it. **Unknown.**
- **The `palantir_models` class** (`GEOTECH_FOUNDRY_CLAUDE_SDK=1`;
  `webapp.palantir_sdk_engine.PalantirClaudeChatModel`). The request types
  the repo uses carry text (`ClaudeChatTextContent`) and images
  (`ClaudeChatImageContent`: base64 PNG, JPEG, GIF or WebP) and nothing else,
  and `_claude_contents` **drops any other block without a word**. A PDF sent
  this way would vanish and the model would answer from the instruction
  alone. Whether the SDK has a document content type is not in the repo.
  **Unknown, and unsafe as the code stands.**

So the design covers both:

1. **`ChatModelEngine` gains a PDF block** (the plain baseline's
   `PdfChatEngine` folded in) and an `accepts_pdf` flag: true for
   `ChatAnthropic` on the direct API; true for the proxy class only once the
   probe passes (`GEOTECH_FOUNDRY_CLAUDE_PDF=1`, set by the brief after the
   probe); false for `PalantirClaudeChatModel` and for the Prompter and OpenAI
   classes.
2. **The SDK class refuses a block it cannot send** instead of dropping it (a
   `ValueError` naming the block). No caller sends one today, so nothing else
   changes.
3. **The pictures route** sends each page as a picture followed by the page's
   text. The picture is rendered at the size the model reads without
   shrinking it (the RI-73 describer's `doc.render(page, budget=...)`); the
   text is what a PDF block would carry, from planlens' frame (the text layer,
   or DI text on a scan), between markers, as the describer sends it. Both
   Foundry routes take images. On a scan with DI text this route sends more
   than the PDF route, since a scanned PDF page has no text layer; the wave
   shows what that is worth.
4. **A probe, two calls** (`report_ingest.model_first.probe_pdf_route`): a
   two-page PDF built in memory by pymupdf, with a code word printed only in
   page 2's text, sent once as a PDF and once as pictures. It reports which
   route returned the word. It ships in the wheel and is the wave's first
   cell.

#### 8.7.1 For the INF-40 brief

- **Run the probe first** (wave MF-0, two calls) on each Claude route the
  workspace has. Set `GEOTECH_FOUNDRY_CLAUDE_PDF=1` only if the proxy
  returned the code word from the PDF.
- **Is Haiku 5.5 imported in the workspace's Models panel?** INF-40's notes
  name only Opus 5.5 and Sonnet 5.5. If it is not, Sonnet describes the pages
  (RI-73: 56 harmful pages against Haiku's 50, at ten times the price over the
  API; on Foundry it is tokens).
- **Bypass the response cache for every repeat arm.** Wave 1 served 12 repeat
  calls from it in under half a second with identical answers, so those
  repeats were not independent. Ask the AI FDE how before the wave.
- **Pass `durable_dir`**, so the run files survive a restart (wave 1 passed
  none).
- **Size the describer's thread pool** for Foundry's tokens-per-minute limit,
  which is what bites there.
- **Name the models per stage** (`stage_models=`): lab and narrative on Opus 5.5;
  logs and calc on Sonnet 5.5, plus the Opus arms of §8.8.

### 8.8 Measurement plan

The same answer sheets and the same scorers (`log_scoring.score_record`,
`lab_scoring.score_record`, `calc_scoring.score_record`,
`narrative_scoring.score_narrative`, with `rescore_saved` bringing both arms
to the current scorers at $0 before any comparison), on Foundry with Opus 5.5
and Sonnet 5.5, as a wave of the INF-40 run package. The local API credit is
spent. Every run's full trace is reviewed (the owner's standing rule), and
the review explains any item that moves.

**Offline first, at no cost:** the grouping score (piece 3: answer-sheet
items that fall whole inside one group, on stage b's saved descriptions and
range labels) and the placement score (piece 5: the 90 % gate of §8.4, on the
saved RI-86 answers). A stage that misses the placement gate gets its
fallback before the wave.

**Blind sets first:** logs blind (9 items), lab blind (15), narrative (8
reports, 6 of them blind). Calc has no blind set (§7 item 4): it moves on its
ten open items (decided, §8.10).

**Wave MF-0, the probe:** two calls per route (§8.7.1).

**Wave MF-1, items handed over** (as RI-86: the pages taken from the answer
sheets). `score_on_cluster`'s reader stages gain `model_first=` and
`stage_models=`.

| Stage | Arms | Models |
|---|---|---|
| logs | today, `on` | Sonnet and Opus (the plain call was measured on Opus only) |
| lab | today, `on` | Opus |
| calc | today, `on` | Sonnet and Opus |
| narrative | today, `on` | Opus (owner 2026-10-25) |

Today's arm is run again on Foundry's Claude in the same wave: every Foundry
number so far is GPT-5.4, and RI-78's Claude numbers came over the direct
API. If the probe passes on one route only, both page routes are run on lab
blind (15 items) to see what the pictures route costs in accuracy.

**Wave MF-2, whole reports:** the `ingest` stage with the switch on and off,
on the stage-b reports that hold blind answer sheets, scored by
`_score_ingest_record` (the reader scorers over the whole record) and by
pages sent to the wrong reader against the hand labels. This is the
measurement that includes finding the pages.

**The acceptance rule.** The switch's default moves from `off` to `on` only
if `on` is at least as accurate as today on each of blind logs, blind lab,
calc (open) and narrative, in both waves:

- per stage, values found (the scorer's total) are at least today's, on the
  same models, the same day and the current scorers; for the narrative,
  recall and precision are both at least today's;
- where the two arms are within one run's noise (the lab repeat moved 14
  values over 11 sheets), both arms run again on the items where they differ,
  and the two runs together decide;
- no item loses values without a reason in the trace review;
- over whole reports, pages sent to the wrong reader are no more than today's,
  and there is no new DIGGS round-trip or schema failure;
- at least 90 % of values on pages with text are placed (§8.4).

The same measurement settles `pdf` against `pictures`.

**Cost per report.** An estimate for the plan's median report (§6: about 150
pages, with 20 narrative, 30 log, 40 lab and 30 calc pages; taken as about 15
logs, 25 lab sheets and 6 printouts), from RI-86's and RI-73's measured costs
per item and per page at Claude API list prices. On Foundry the bill is the
organisation's, and the number to watch is tokens.

| Step | Model | Calls | Estimate |
|---|---|---|---|
| describe every page | Haiku | 150 | $0.37 |
| label by sections | Opus | 1 | $0.30 |
| triage, bound identity | as today | 1-3 | $0.05 |
| logs | Sonnet | about 15 | $0.85 |
| lab | Opus | 5-25 | $3.00 |
| calc | Sonnet | 6-8 | $0.40 |
| narrative | Opus | 1-2 | $0.60 |
| **total** | | **about 190** | **about $5.60** |

That is about 1 M input and 0.2 M output tokens. Today's pipeline on the same
models comes to about $7.70 for the same report: its readers cost 1.3 to 2.6
times the plain call per item (RI-86's and RI-78's runs), and the vision pass
and the label review add about $0.50. Lab on Opus is more than half of the
model-first cost. The wave's own token counts replace this table.

### 8.9 Build pieces, in order

Each piece is one builder and one reviewer. A model-facing piece changes what
the model is asked; it sits behind the switch and is measured before the
default moves. RI-73 (wip/ri73) merges first. After it, pieces 1, 2, 3 and 5
can be built offline in any order; 4 and 6 come before 7; 8 and 9 come last,
before the wave.

1. **The PDF block, the SDK guard, the probe** (S). `report_ingest/engine.py`
   (`ChatModelEngine`: a `pdf` block, `accepts_pdf`),
   `webapp/palantir_sdk_engine.py` (`_claude_contents` refuses a block it
   cannot send), `report_ingest/model_first.py` (new: `probe_pdf_route`),
   `module_work/report_ingest_harness/run_plain_baseline.py` (uses the
   engine's block); tests in `report_ingest/tests/test_chat_model_engine.py`
   and `webapp/tests/test_palantir_claude_engine.py`. Not model-facing.
2. **Sending the pages** (S). `report_ingest/model_first_pages.py` (new):
   `cut_pdf` from the baseline, the API limits, the pictures route (the
   picture and its text between markers), the sentence mapping PDF pages to
   report pages. Tests on a PDF made inside the test. Not model-facing on its
   own: it carries what piece 7 asks.
3. **Grouping the pages for each call** (S-M).
   `report_ingest/model_first_pages.py`: logs by identifier and `continues`,
   lab windows cut at the descriptions' sheet starts, calc and narrative as
   today. Scored offline on stage b's saved descriptions and range labels.
   Tests on made-up descriptions. Not model-facing.
4. **Schema additions and instructions** (M, **model-facing**).
   `report_ingest/model_first.py`: the subclasses of §8.5, the per-stage
   instructions moved from `run_plain_baseline.INSTRUCTIONS` with a version
   string, and the builder's mapping of the new fields into the record.
   Tests: today's reading classes give the same `strict_schema` output (a
   snapshot); the subclasses carry the new fields.
5. **Placement by text match** (M). `report_ingest/locate.py` (new: the page
   check, the anchors, the words, the ruler row, the QA note) and its tests;
   then the offline placement score on the saved RI-86 answers, per stage,
   against the 90 % gate. Not model-facing.
6. **Builders in flag mode** (M). `report_ingest/log_reader.py`,
   `lab_reader.py`, `calc_reader.py` (a `checks` parameter, default `refuse`;
   the PDF-local page fix; each lab test's pages narrowed) and
   `narrative_reader.py` (the caption count as a QA note under `on`), with
   their tests. Not model-facing.
7. **The model-first readers** (M, **model-facing**).
   `report_ingest/model_first.py`: one call per group, the cut-off halves,
   build, place, the result dict in today's shape; the narrative's parts and
   summary call. Tests with a scripted engine.
8. **Graph wiring and the byte-identical test** (M).
   `report_ingest/graph.py` (`model_first=`, `stage_engines=`,
   `page_route=`; the vision voter and the review off under `on`; the floors
   not run; the `items/mf-` keys; bound documents) and
   `report_ingest/tests/test_graph*.py` (switch off: every request and output
   file byte-identical to master's; no model-first module imported). Not
   model-facing beyond pieces 4 and 7.
9. **Measurement plumbing** (S). `report_ingest/cluster_scoring.py`
   (`model_first=`, `stage_models=`, `page_route=` on the reader stages and
   `ingest`; run files and `RESULTS.md` name the setting),
   `module_work/report_ingest_harness/run_plain_baseline.py` (scores through
   the same path), tests in `report_ingest/tests/test_cluster_scoring.py`. The
   wave brief itself (INF-40, §8.7.1) follows.

**Fallback, built only if piece 5 misses the 90 % gate in a stage** (S-M per
stage): the floor as a source of boxes for that stage (§8.4).
`report_ingest/floor.py` (`settle` gains a `place` policy, default today's)
and that stage's `log_floor.merge_investigations`,
`lab_floor.merge_lab_tests` or `calc_reader.merge_calculation`, with tests
that the default reproduces today's merges on the saved Foundry and local
runs. Not model-facing.

### 8.10 Decisions, and what the simple path gives up

Decided here (they were owner questions in the first draft):

1. **Calc moves on the open set's numbers.** It has no blind answer sheets
   (§7 item 4). Under `on` there is no calc prompt tuned on those ten items:
   the instruction is RI-86's short general one and the answer schema is
   today's, so the open-set bias is small. What bias remains (the kind
   vocabulary, the schema and the scorer's RI-74 fixes were all written while
   reading these items) applies to both arms alike.
2. **No code-read value enters the record under `on`.** There is no fill
   setting; the schema additions (§8.5) close the gaps the floor filled, and a
   value only the code read is a QA note at most.
3. **The range labels replace the vision voter and the label review** under
   `on`.

What the simple path gives up, as far as the numbers say:

- **The floor's added log values.** On the pipeline's own model reading the
  floor and merge add 9 open and 15 blind log values (287 to 296, 167 to 182),
  and 10-16 per arm in CLAUDE LOCAL RUN 1; on lab they add 0 and on calc -1.
  The plain arm is 7 blind and 17 open log values behind the pipeline. The
  schema additions give a slot to about 21 blind values the plain schema
  cannot hold (the project fields, 12 misses in every pipeline arm; recovery
  lengths, 9 on R34_p49 and R21_p96). Whether the model fills them is what wave MF-1
  measures. A value the model leaves out is now invisible in production: there is no floor to
  catch it and no note to say so.
- **The second voter's flags.** Today every split between the floor and the
  model is a `disagreement` entry, the trigger for a second look (calc kept
  27 and 36 of them per arm in Foundry run 4, after the place rule). Under
  `on` there are none; the per-value check left is "not found printed on the
  cited page", which catches an invented number but not a printed number put
  in the wrong slot.
- **Boxes are unmeasured** until piece 5: today 557 of 619 log values (90 %)
  and 71 of 72 lab tests are placed. A layer boundary the log draws but does
  not print gets its object's box, not one of its own. A scan with no DI text
  gets its pages and no boxes, as today's floor would.
- **The narrative's caption counts** stop overriding the model's answer;
  unmeasured either way.
- **Grouping without planlens' item key and the lab floor's specimen starts**
  rests on the descriptions; the offline score (piece 3) says whether either
  has to come back.

Other risks: small samples (9 blind logs, 15 blind lab sheets, 8 narratives,
10 open calc items, one run per arm, answer sheets made by Claude sessions,
RI-72); the pictures route is unmeasured (the plain result is the PDF
route's); a dense lab window may stop at the output cap (the halves rule
covers it at one more call).
