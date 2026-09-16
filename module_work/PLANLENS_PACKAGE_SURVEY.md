# PLANLENS PACKAGE SURVEY — PARKED 2026-09-14 (owner: pivoting to GSE app work)

**Status 2026-09-16 (final): SHIPPED. Part A (stored scale, layers, fill), Tier 1
(fuzzy search, quantities, duplicate scans) and the MCP server are in planlens 0.4.0
(PyPI); the app wiring is in geotech-staff-engineer 5.18.0 (PyPI). Still TODO:
Tier 2 scanned-page engines (pending Funhouse confirmation on vendored models), A4
revision comparison, and the toolkit MCP server (see FUTURE_IDEAS).**
Original status line: PARKED, not approved, nothing built. Owner's instructions on parking:
(1) it is a todo, not a train; (2) **it must be translated into plain English
before the owner reviews it** — the owner is not a developer and the text below
is written for an agent. A plain-English version means: what each item would let
a reviewer DO with a document, what it costs (install size, cluster approval,
model files), and which items are blocked by the organisation's package
firewall — no package names in the headline, no acronyms unexplained.

**Compliance status (owner asked 2026-09-14 "was this checked against the banned
list?"): NO.** The organisation's list (the Nexus approved/blocked inventory the
SDK's `nexus_pip_install` loads) returns 404 from outside the enclave and no copy
exists locally; the DT violation list from 2026-09-02 was never saved. Only
public license + advisory data was checked (all clean). Part E below has the
gate every new pin must pass on the cluster first. A second finding from that
check: the cluster's egress is firewalled, so every candidate that downloads
model weights at runtime is unusable as published.

Original plan file: `C:/Users/socon/.claude/plans/please-scour-the-interwebs-splendid-moler.md`
(this is a verbatim copy, kept in the private app repo — never into planlens' public repo).


## Plain-English summary (written for the owner, 2026-09-16)

**What this is about.** planlens is the part of the app that reads a PDF
submittal so the AI can review it: which pages are prose, which are drawings,
which are boring logs; the text with its position on the page; the tables; the
reviewer's Bluebeam comments; the CAD text hidden behind stroked lettering. I
went looking for ready-made tools that would let it read more, and checked how
each fits our situation.

**The biggest improvements need nothing new installed.** The PDFs already
contain three things we throw away:

1. *The scale.* When a drafter or reviewer calibrates a sheet in Bluebeam or
   Acrobat and saves it, that calibration is stored inside the PDF, and so is
   every measurement markup. Today the app has to read the scale off the title
   block or be told. Reading the stored calibration gives it the drafter's own
   number. This attacks the most dangerous error class we have: a plausible
   number in the wrong units, with nothing to flag it.
2. *Layer names.* A PDF plotted from CAD keeps its layer names. The CAD-file
   side of planlens already uses them; the PDF side ignores them. "Existing
   conditions" versus "proposed" versus "dimensions" is often just the layer.
3. *Filled shapes.* Solid boring dots, solid arrowheads, hatching. Right now
   they look the same as outlines. Filled circles are how borings are drawn.

A fourth item, comparing two revisions of a drawing the way Bluebeam's Compare
Documents does, can also be built with what we already have.

**Small, low-risk add-ons.** Three little libraries, each doing one job:

- *Forgiving search.* Text read optically or from stroked lettering has letter
  errors. A search for RAKER should still find RAKEB.
- *Numbers out of the narrative.* Pull "borings at approximately 40-foot
  centers" out of the report text as a number with a unit and a page location,
  so the app can compare what the report says with what the plan measures.
  That comparison is where real findings come from.
- *Duplicate-page detection for scans.* Also helps line up pages between two
  revisions.

**Medium effort: reading scanned pages better.** Tables on scanned boring logs,
and page layout (headings, figures, captions) on scans, can be handled by small
AI models that run on an ordinary CPU. The catch: these tools fetch their model
files from the internet the first time they run, and the cluster has no
internet. We would have to package the model files ourselves, which is exactly
what was done for the current OCR engine. So these are candidates, not installs.

**Strategic: a standard plug.** There is now an industry-standard way for AI
assistants to connect to tools (called MCP). Exposing the document-review tools
through it would let anyone with a supporting assistant (Claude Desktop, Claude
Code, and others) review a PDF with planlens without our web app. That is a
strong story for the banner-app pitch, and it is a small amount of code.

**Ruled out, and why.** Two popular PDF layout tools are licensed for
non-commercial use only, so an organisation cannot use them. One well-known
OCR family restricts its model files. Several others need heavy
machine-learning frameworks. One thing to know rather than act on: the core PDF
library we already depend on has a license (AGPL) that corporate license
scanners sometimes flag. It is normally fine for internal use. Worth one
question to DT before TinyApps, not a reason to switch.

**The security check, honestly.** None of this has been checked against the
organisation's banned-package list. That list is a web page inside the enclave
that I cannot reach from your laptop, and the list DT sent earlier this month
was never saved. What I could check were the public vulnerability databases,
and everything came back clean, but that is only half of what the firewall
looks at. The rule going forward: before anything is added, run the
organisation's own install helper on the cluster for each package; it says
whether the package is quarantined. And ask the Funhouse developer for a copy
of the list so it can be checked from the laptop in future.

**Recommended order.** Step one (nothing to install, biggest value, safest).
Then the small add-ons, after the cluster check. Then the scanned-page work
only if you want it and after the model files are packaged. The standard plug
whenever the pitch needs it.

---

# planlens capability boost — package survey and adoption plan

**Date:** 2026-09-14. planlens is at 0.3.0 (PyPI, MIT). App pin `planlens[raster]>=0.3`.
Owner rules that govern this plan: sequential work, one subagent at a time, planlens
work burns usage so spikes must be small and measured; prefer free over paid; never a
required paid dependency; planlens repo is PUBLIC (no field-feedback names/paths in it).

## Context

planlens turns AEC PDFs into review-ready data for an LLM: page map + structure,
located text, tables (`find_tables`), review markups, hidden CAD text, search, render,
and the drawing IR (leaders/dimensions/bubbles) with `measure`/`spatial`. Its own
design notes list what is NOT built (`planlens/document/DESIGN.md` "Not built yet",
HANDOFF §0a "Open, in order"):

- RapidOCR as a document text source; the `[ocr]` extra conflicts with `[raster]`
  (rapidocr hard-requires GUI `opencv-python`; headless deploys use a `--no-deps` recipe)
- roles/headings from the text layer; multi-column reading order; figure + caption
  detection; tables spanning pages; visual reading order on form pages (boring logs)
- tables inside scanned images (`find_tables` is vector-only)
- a multi-firm TEXT-BEARING eval set; scanned-drawing behaviour unmeasured

Local checks made for this survey (all confirmed by grep, 2026-09-14):

- `Document.search` (`planlens/document/document.py:440`) is substring/regex only —
  no fuzzy match, so OCR'd or SHX-stroked text ("RAKEB" for "RAKER") misses.
- No reading of PDF layers (optional content): the `layer` key that
  `page.get_drawings()` returns since PyMuPDF 1.22 is unused anywhere in planlens.
- No reading of PDF measurement data (`/VP` viewports, `/Measure` dictionaries) —
  the calibrated scale Bluebeam/Acrobat store in the file is ignored.
- `fill` never appears in `planlens/ir/ingest.py` — the "fill dropped by extractor"
  item from the 09-07 plan is still open.
- `duplicate_of` in the page map keys on text + path count, so it is blind on scans.

The survey below is what the web offers against those gaps, verified on PyPI
metadata (versions, licenses, `requires_dist`) on 2026-09-14.

**Owner question (2026-09-14): "Has this been reviewed for security against the
list of banned packages?" — No. See Part E for what was and was not checked, why
the org's list is unreachable from this machine, the egress blocker it surfaced,
and the gate now required before any pin is added.**

## Part A — Findings that need NO new package (do these first)

These are PDF-spec / PyMuPDF features already available under the current dependency.
Each is one small module plus tests, measured on the real submittal.

| # | Capability | Mechanism | Why it matters |
|---|---|---|---|
| A1 | **Calibrated page scale from the file** | ISO 32000 §12.9: page `/VP` array of Viewport dicts, each with `/Measure` `/Subtype /RL` (`/R` ratio string e.g. "1 in = 20 ft", `/X`/`/Y`/`/D`/`/A` number-format arrays). Bluebeam "Store Scale in Page" and Acrobat's measuring tools write these; Bluebeam measurement markups (`/IT /LineDimension`, `/PolyLineDimension`, `/PolygonDimension`) carry their own `/Measure`. Read via `doc.xref_get_key(page.xref, "VP")` / `annot.xref`. | `measure.py` refuses to turn points into feet without a resolved scale — today the caller resolves it by reading the scale note. A viewport gives the drafter's own calibration, per page and per region, at confidence 1.0. Also unlocks: reading Bluebeam measurement markups as data (lengths/areas the reviewer already measured). |
| A2 | **PDF layers as IR layer** | `page.get_drawings()` → `layer` key (OCG name), `doc.get_ocgs()` / `get_layers()` for visibility. CAD→PDF plots keep layer names (`C-TOPO`, `S-BORING`, `A-ANNO-DIMS`). | The DXF leg already resolves layers; the PDF leg reports none. Layer names are the cheapest classifier of what a line IS (a dimension layer vs. an existing-conditions layer) and let `slice` queries filter by layer on PDFs too. |
| A3 | **Fill from `get_drawings`** | Each path dict has `fill`, `fill_opacity`, `stroke_opacity`, `dashes`, `width`. | Solid arrowheads, hatch, filled symbols (boring markers!) are currently indistinguishable from outlines. Directly serves the boring-spacing slice (filled circles = borings). |
| A4 | **Text-layer diff / revision compare** | PyMuPDF render + OpenCV (already deps): ECC/phase-correlation alignment, thresholded diff mask, red overlay; `difflib` on `TextLine`s per page for the text half. | "What changed between Rev B and Rev C" is a core review question. No library worth adopting exists (see Part D), so this is a planlens feature, not a package. |

## Part B — Packages recommended, ranked

Legend: **Lic** = license, **Py** = minimum Python, **cv2** = which OpenCV build the
package pins (the namespace trap: `[raster]` uses `opencv-python-headless`; anything
that pins the GUI build needs the README's `--no-deps` recipe on headless deploys).

### Tier 1 — small, permissive, no model downloads; each is one targeted lever

| Package | Ver / Lic / Py | Unlocks in planlens | Where it plugs in | Caveats |
|---|---|---|---|---|
| **rapidfuzz** | 3.14.6 / MIT / ≥3.11 (3.9.x line supports 3.10) | Fuzzy `search` (OCR + SHX text), near-duplicate line matching, matching prose mentions to drawing labels for reconciliation | `Document.search(fuzzy=True, score_cutoff=…)`; `spatial`/reconciliation helpers | C++ wheel, no deps. Pin `rapidfuzz>=3.9` so 3.10 still resolves. |
| **quantulum3** | 0.10.0 / MIT / — | Quantities with units out of narrative ("borings at approximately 40-foot centers", "6300 mm", "21°") → the input side of narrative-vs-drawing reconciliation | new `planlens.document.quantities` → list of (value, unit, span, line id, page) | Deps `inflect`, `num2words` (pure Python); skip the `[classifier]` extra (scikit-learn). Will NOT parse feet-inches (`7'-6"`) or stations (`10+50`) — planlens needs its own regex for those anyway. Measure precision on the submittal's calc pages before trusting. |
| **imagehash** | BSD / — | Perceptual duplicate pages on scans; page matching between two revisions (A4) | `structure.py` duplicate detection; `compare` | Deps `Pillow`, `numpy`, `scipy`, `PyWavelets` — small. |
| **shapely** (already a dep of the OCR leg) | BSD | `STRtree` spatial index for IR queries on 8k-entity sheets; `concave_hull` for `spatial.py` footprint area (today flagged, not computed) | `ir/queries.py`, `ir/spatial.py` | `spatial.py` is numpy-only by design — keep shapely optional (import-guarded) if adopted. |

### Tier 2 — OCR / layout on onnxruntime (matches the no-torch stance)

**Read Part E first: every package in this tier downloads its model weights at
runtime, and the cluster's egress is firewalled. None of them works on the cluster
or on TinyApps as published; each needs its ONNX models vendored (planlens package
data or a private wheel through Nexus) — the way `planlens.ocr` already ships
PP-OCR models in-wheel. Treat this tier as "candidate engines", not installs.**

| Package | Ver / Lic / Py / cv2 | Unlocks | Where it plugs in | Caveats |
|---|---|---|---|---|
| **onnxtr** `[cpu-headless]` | 0.9.0 / Apache-2.0 / ≥3.11 / **headless** | A second `planlens.ocr` backend with NO cv2 conflict; orientation detection + page straightening; word boxes + confidence; DBNet/FAST detectors, CRNN/PARSeq recognisers; 8-bit CPU models | `planlens.ocr` behind a backend enum; the Document text-source protocol (the RapidOCR-as-text-source item) | Models download at runtime from Hugging Face Hub (cluster/TinyApps proxy may block — verify). Deps include `pypdfium2`, `rapidfuzz`, `huggingface-hub`. |
| **wired_table_rec / lineless_table_rec / table_cls** (RapidAI TableStructureRec) | 1.2.0 / Apache-2.0 / — / **GUI** `opencv-python` | Table STRUCTURE on scanned images: cell boxes, logic coordinates, HTML; accepts RapidOCR boxes as input `[[score, box, text], …]`; ~100 ms wired / ~500 ms lineless per table | `tables.py` scanned-page path (today `find_tables` is vector-only); boring logs on scanned appendices | Runtime model download; GUI cv2 pin (same `--no-deps` recipe as rapidocr); deps `scipy`, `scikit-image`, `Shapely`. |
| **rapid-layout** | Apache-2.0 / — / **GUI** `opencv-python` | Page regions: title / text / figure / table / header / footer on scans and text pages → fills "roles/headings" and "figure + caption detection" | `classify.py` evidence + a `regions` field on `PageSummary`; caption ↔ figure pairing | Offers PP-DocLayout (Apache) AND DocLayout-YOLO weights (AGPL lineage) — use the PP-DocLayout family only; confirm each model's license in the RapidLayout docs at install time. Runtime download. |
| **img2table** | 2.0.0 / MIT / — / **GUI** `opencv-contrib-python` + `pypdfium2` | Alternative ruling-line table extractor on images AND PDFs with cell bboxes; `[rapidocr]` extra; borderless tables | Same slot as TableStructureRec | Pulls the contrib GUI build. Overlaps Tier-2 row 2 — run a bake-off on the boring-log pages and keep ONE. |

### Tier 3 — strategic / heavier; owner-gated

| Package | Ver / Lic | Unlocks | Notes |
|---|---|---|---|
| **mcp** (official MCP Python SDK, FastMCP server) | v2 line / MIT | `planlens-mcp`: the ReviewToolkit as an MCP server (stdio / streamable HTTP) → usable from Claude Desktop, Claude Code, Cursor, and the app via `langchain-mcp-adapters` (deepagents supports MCP tools). | The specs are already framework-neutral JSON-Schema, so this is ~150 lines. It is the strongest DISTRIBUTION lever for the "banner TinyApp" pitch: any engineer with an MCP host reviews their own PDF without the Streamlit app. |
| **docling** | 2.127.0 / MIT / ≥3.10 | Full layout model (reading order, section headings, captions, tables spanning pages) and TableFormer; RapidOCR integration | Torch is unavoidable for the layout/table models (`docling-slim[standard]` pulls `torch`+`torchvision`; the `models-onnxruntime` extra only accelerates, it does not replace them — verified from `requires_dist`). CPU wheels are hundreds of MB. Only as an optional `LayoutSource` behind the same protocol as `AzureLayout`, and only if Tier 2 layout proves insufficient. Too heavy for TinyApps. |
| **python-docx 1.2** / **docx2python** | MIT / MIT | Word specs and reports with COMMENTS (python-docx 1.2 added a comments API; docx2python extracts comments, headers, footnotes) | Scope expansion beyond PDF. Owner: "PDF nearly always" → park until a .docx shows up in field feedback. |
| **opencv-contrib-python-headless** (FastLineDetector), **sknw**, **vtracer** | Apache / BSD / MIT | Better raster linework on scans: FLD (≈10× faster than the removed LSD), skeleton→graph (`sknw`), raster→SVG (`vtracer`, built for gigapixel blueprint scans) | Arrowhead/raster tuning is FROZEN by owner decision. Park until a scanned-drawing eval set exists. Swapping headless→contrib-headless is namespace-safe (superset). |

## Part C — Licensing facts the owner should know (no action proposed, just facts)

- **pymupdf-layout is Polyform NONCOMMERCIAL (or Artifex commercial).** It is the
  "optional layout package" whose advisory `tables.py` already silences. **pymupdf4llm
  ≥1.28 now REQUIRES pymupdf-layout**, so `pip install pymupdf4llm` pulls a
  noncommercial-licensed component. Neither is usable for office deployment without an
  Artifex license. Do not adopt; do not let a drive-by `pip install pymupdf4llm` in.
- **PyMuPDF itself is AGPL-3.0** (dual-licensed by Artifex). planlens being MIT does
  not change PyMuPDF's terms; the whole document layer sits on it. Internal office use
  is generally fine, but DT's Nexus license sweep (the one that flagged matplotlib
  "License-Threat Not Assigned") may flag AGPL. The permissive fallback is
  **pypdfium2** (Apache-2.0 / BSD-3, Google's PDFium; used by onnxtr, img2table,
  docling-parse) + **pypdf** (BSD) for dictionaries + **pdfplumber** (MIT) for
  char/line attributes — but the switching cost is the entire document layer and the
  display-list rotation work. Recommendation: ask DT once, before TinyApps; do not
  switch pre-emptively.
- **Surya / marker** (datalab): code Apache-2.0 but model weights are a modified
  Open RAIL-M / non-commercial above a revenue threshold. Not free for a firm. Skip.
- **MinerU** (AGPL-3.0), **DocLayout-YOLO** (AGPL via ultralytics), **PaddleOCR**
  (needs `paddlepaddle`, no Python 3.14), **Tesseract**-based tools (system binary),
  **unstructured** (heavy), **cloud parsers** (paid — owner rule). Skip.
- **rapidocr 3.9.2 still pins GUI `opencv_python`**; RapidAI issue #737 (make the
  build selectable) is OPEN, milestoned v3.10.0. When it ships, the `--no-deps` recipe
  can go. planlens' `[ocr]` extra pins the legacy `rapidocr-onnxruntime<1.3`; the
  maintained line is unified `rapidocr` 3.x — moving is a known item, unchanged here.

## Part D — Looked at and rejected (so nobody re-derives it)

pdf-comparison-tool (CLI only, 1 commit, no license), pdf-diff (needs poppler
binaries), kreuzberg (nothing verifiable found), layoutparser (stale), deepdoctection
(heavy), camelot (vector-only, overlaps `find_tables`), pdfplumber (would duplicate
what PyMuPDF already gives, slow on D-size sheets), CLIP/image embeddings (rejected
earlier for charts, same reasoning), VLM OCR models (GOT-OCR2, olmOCR, dots.ocr,
PaddleOCR-VL — GPU-class, the app's own vision model already does "look").

## Part E — Compliance against the CfA Nexus firewall and the DT sweep

**Straight answer: this survey was NOT checked against the organization's
banned/blocked package list, and it cannot be from this machine.** What the check
has to consult, and where each source stands:

| Source of truth | Where it lives | Status from here (2026-09-14) |
|---|---|---|
| **Nexus approved/blocked inventory** (`nexus_approved_blocked.csv` / `.html` on the Funhouse static site) — the list `nexus_pip_install` loads before every install | SDK reference tree, `funhouse/utils/package.py:149` | **HTTP 404** for the CSV, the HTML and the site root. The SDK's own `docs/access-requests.md` already records the link as broken. No local snapshot exists under `dev/`, Downloads, or memory. |
| **DT package-violation list** (2026-09-02: groundhog HIGH, pandas 2.x critical, matplotlib "License-Threat Not Assigned", tornado) | Pasted into a session; never saved to disk. Commit `d458d7f` and the memory note carry only the highlights. | Not available. |
| **Nexus firewall quarantine** (403 "Requested item is quarantined") | Observable only on the cluster at install time. "No waivers are available for this project" (`docs/DATABRICKS_INSTALL.md` §10). "A quarantined package means a CVE" (SDK funhouse-gotchas skill). | Can only be probed on-cluster. |

**What WAS checked (a proxy, read-only):** license per PyPI metadata, and known
security advisories per exact version via Google deps.dev (OSV / GitHub advisory
feed). That is the CVE half of what a Sonatype-style firewall evaluates; it says
nothing about the org's license-threat groups, malware heuristics, or manual
blocks. Transitive dependencies were not individually checked.

| Package | Version checked | License | Known advisories |
|---|---|---|---|
| rapidfuzz | 3.14.6 | MIT | none |
| quantulum3 | 0.10.0 | MIT | none (deps `inflect`, `num2words` not checked) |
| imagehash | 4.3.2 | BSD-2-Clause | none |
| onnxtr | 0.9.0 | Apache-2.0 (deps.dev: "non-standard") | none |
| wired-table-rec | 1.2.0 | Apache-2.0 | none |
| rapid-layout | 1.2.1 | Apache-2.0 | none |
| rapid-table | 3.0.2 | Apache-2.0 (PyPI) | **not version-checked** |
| img2table | 2.0.0 | MIT | none |
| docling | 2.127.0 | MIT | none (torch stack not checked) |
| mcp | 2.2.0 | MIT | none |
| python-docx | 1.2.0 | MIT | none |
| docx2python | 3.7.1 | MIT | none |
| pypdfium2 | 5.13.0 | Apache-2.0 / BSD-3 (deps.dev: "non-standard") | none |
| *baseline* pymupdf | 1.28.2 | AGPL-3.0 (deps.dev: "non-standard") | none |
| *baseline* rapidocr-onnxruntime | 1.2.3 (planlens pin) | Apache-2.0 | none |
| *baseline* rapidocr | 3.9.2 | Apache-2.0 | none |

**New blocker found while checking: the cluster's egress is firewalled** ("packages
from CfA Nexus only; egress firewalled — API-First gateway is the only
external-data route", environment facts recorded 2026-09-01). Every Tier-2
candidate fetches model weights at runtime — onnxtr from Hugging Face Hub;
wired_table_rec / lineless_table_rec / table_cls by auto-download (since
2025-03); rapid-layout; rapid-table. They will fail on the cluster and on TinyApps
even if Nexus admits the wheel. RapidOCR was chosen precisely because its PP-OCR
models ship inside the wheel. Tier 2 is therefore viable only with **model
vendoring**: download the ONNX files on the low side, confirm each model's license
(PP-* models Apache-2.0; DocLayout-YOLO weights AGPL — excluded), measure sizes,
and ship them as planlens package data or as a private wheel through Nexus (the
SDK's `NexusPackageManager.upload_package` exists for that; the Nexus admins own
the hosted repo).

**The gate this plan now requires before ANY new dependency is pinned:**

1. On the cluster, scratch notebook, one package at a time, unpinned:
   ```python
   from funhouse.utils import nexus_pip_install
   nexus_pip_install("rapidfuzz")
   ```
   Record the reproducible pin line it prints and every version it reports as
   quarantined. This is the org's own tool and the only authoritative check.
2. Import-and-run smoke in that same env with egress as-is — proves there is no
   runtime download.
3. Only then add the pin to `pyproject.toml` (as an optional extra), and record
   the probe result in `docs/DATABRICKS_INSTALL.md` §7 and §11.
4. Ask the Funhouse dev for a current copy of `nexus_approved_blocked.csv` and
   commit it under `module_work/nexus/` with a small `check_pins.py` that fails
   when any pin in either `pyproject.toml` names a blocked version. That is the
   durable fix, it is what the SDK's own docs recommend, and it lets the check run
   locally instead of on the cluster.

What this means for ordering: Part A (no new packages) is unaffected. Tier 1 is
pure Python/C++ with no downloads and no advisories and can be built locally now,
but nothing gets pinned until step 1 passes. Tier 2 waits on model vendoring plus
step 1. Tier 3 (MCP) is also download-free and follows the same gate.

## Recommended execution (if approved): one slice at a time, measured

Owner can stop after any step. Each step is a small commit on a planlens feature
branch with tests; nothing bumps the version until the owner says so.

**Step 1 — Part A on the current deps (no install).**
- A1 viewports/measure: `planlens/document/scale.py` → `PageSummary.viewports`
  (bbox, ratio string, x/y unit factors, source `viewport|measurement_markup`), and
  a `resolve_scale(page)` helper `measure.py` can consume. Markups of `/IT
  *Dimension` gain `measure` (value, unit) in `annotations.py`.
- A2 layers: `ir/ingest.py` PDF leg carries `layer` from `get_drawings`; `slice`
  filters already accept `layer`.
- A3 fill: `ir/ingest.py` keeps `fill`/`fill_opacity`; `Polyline`/`Circle` gain
  `filled: bool`; `queries.py` arrowhead/bubble finders can read it (no re-tuning —
  just carried as evidence; tuning is frozen).
- Measure on: the private submittal PDF (local path under
  `GeotechStaffEngineer/module_work/field_feedback/2026-09-09_nairobi-soe_v5.11.2/raw/files/`)
  and the ten Mecklenburg sheets
  (`GeotechStaffEngineer/module_work/drawing_ground_truth/mecklenburg/*.pdf`).
  Report: pages with viewports, ratio strings found, OCG names per sheet, count of
  filled paths. `doc_claims_check.py` must still reproduce every published figure.

**Step 2 — Tier 1 spike (rapidfuzz + quantulum3), optional extra `[text]`.**
- `Document.search(fuzzy=True)` over lines + hidden CAD text + OCR text; return
  `score` per hit. Test: the submittal's SHX callouts with one substituted letter.
- `planlens.document.quantities`: quantulum3 + a feet-inches/station regex; every
  hit carries page, line id, bbox. Test: calc pages and the transmittal narrative;
  precision/recall on 30 hand-checked mentions.
- Then the first reconciliation tool: `find_quantity_mentions(unit=…)` → a candidate
  list the model compares against `find_dimensions` output. This is the boring-spacing
  slice's missing half.

**Step 3 — Tier 2 bake-off in a SCRATCH venv (never alongside the gate). Local
evaluation only; adoption requires vendored models and the Part E gate.**
- `pip install "onnxtr[cpu-headless]"` in a headless-only venv → confirm the cv2
  namespace stays headless; run on the corpus OCR validation sheets; compare against
  the committed `ocr_coverage_check` numbers (88–100 % coverage, 2.2–15.5 pt median).
  Adopt as `planlens.ocr` backend `"onnxtr"` only if it matches or beats RapidOCR AND
  its detector + recogniser ONNX files can be vendored (license, size) — runtime
  Hugging Face downloads are blocked on the cluster.
- Scanned-table bake-off: `wired_table_rec` vs `img2table[rapidocr]` on the boring-log
  pages rendered at 200 dpi; metric = cell count and text-in-cell match vs
  `find_tables` on the vector original. Keep one.
- `rapid-layout` (PP-DocLayout weights only) on 20 text/figure pages; metric =
  heading/caption/figure regions vs hand labels. Feeds `classify.py` evidence.

**Step 4 — MCP server (owner-gated; strategic).**
- `planlens/mcp_server.py` with FastMCP: one tool per `ReviewToolkit` spec, images
  returned as MCP image content. Verify from Claude Code (`claude mcp add`) on the
  synthetic submittal fixture, then the real one.

Deferred / parked (owner decision): docling as an optional `LayoutSource`; Word
document support; raster vectorization; the PyMuPDF licensing question to DT.

## Verification

- Every step: `cd planlens && pytest planlens -q` (875 today) and
  `GeotechStaffEngineer/.venv/Scripts/python -m pytest funhouse_agent/deep/tests/test_document_tools_offline.py -q`.
- Corpus figures: `GeotechStaffEngineer/module_work/drawing_ground_truth/doc_claims_check.py`
  must reproduce every number the README publishes after A2/A3.
- New optional extras install cleanly in a fresh HEADLESS venv with `[raster]`
  present (`python -c "import cv2; print(cv2.__file__)"` shows one distribution).
- Nothing from the private submittal (names, project numbers, paths) enters the
  planlens repo; measurements are recorded in the app repo's field_feedback ledger.
- Run the full release gate only when nothing else heavy is running (it was killed for
  memory once on 2026-09-14).

## Sources

- pymupdf-layout license: https://pypi.org/project/pymupdf-layout/ ; pymupdf4llm: https://pypi.org/project/pymupdf4llm/
- PyMuPDF licence discussion: https://github.com/pymupdf/PyMuPDF/discussions/971 ; pypdfium2: https://github.com/pypdfium2-team/pypdfium2
- PyMuPDF optional content / `layer` key: https://pymupdf.readthedocs.io/en/latest/recipes-optional-content.html
- PDF 1.6 reference §8.7.5 / ISO 32000 §12.9 measure dictionaries: https://opensource.adobe.com/dc-acrobat-sdk-docs/pdfstandards/pdfreference1.6.pdf ; Bluebeam "Store Scale in Page": https://support.bluebeam.com/user-manual/menus/window/measurements-panel.html
- OnnxTR: https://github.com/felixdittrich92/OnnxTR ; https://pypi.org/project/onnxtr/
- RapidOCR opencv issue #737: https://github.com/RapidAI/RapidOCR/issues/737 ; rapidocr: https://pypi.org/project/rapidocr/
- RapidAI TableStructureRec: https://github.com/RapidAI/TableStructureRec/blob/main/README_en.md ; RapidLayout: https://github.com/RapidAI/RapidLayout ; rapid-table: https://pypi.org/project/rapid-table/
- img2table: https://github.com/xavctn/img2table
- docling install / OCR engines: https://docling-project.github.io/docling/getting_started/installation/ ; https://pypi.org/project/docling/
- Surya model license: https://github.com/datalab-to/surya/blob/master/MODEL_LICENSE
- quantulum3: https://github.com/nielstron/quantulum3 ; rapidfuzz: https://pypi.org/project/rapidfuzz/
- MCP Python SDK: https://github.com/modelcontextprotocol/python-sdk ; deepagents + MCP: https://docs.langchain.com/oss/python/deepagents/overview ; langchain-mcp-adapters: https://deepwiki.com/langchain-ai/langchain-mcp-adapters
- python-docx comments: https://python-docx.readthedocs.io/en/latest/user/comments.html ; docx2python: https://github.com/ShayHill/docx2python
- vtracer: https://github.com/visioncortex/vtracer ; FastLineDetector: https://docs.opencv.org/3.4.2/df/d4c/classcv_1_1ximgproc_1_1FastLineDetector.html
- pdf-comparison-tool (rejected): https://github.com/darkrubiks/pdf-comparison-tool
