# `report_ingest` — a geotechnical report as organised, cited data

Part of the report-ingest train (`module_work/REPORT_INGEST_PLAN.md`). planlens
turns a PDF into pages with kinds, located text, an outline and a first draft of
what each page **is**; this package adds the geotechnical judgement and the
record. The lab reader, the narrative reader and the reconciler are later work
packages.

| Piece | Module | Shape |
|---|---|---|
| 0b. Document triage | `triage.py` | one structured call |
| 0c. Label review | `label_review.py` | an agent loop with four tools |
| The record | `model.py` | pydantic; the product everything else exports |
| 2. Log reader | `log_reader.py` | one structured call per log, image alongside |
| 4. DIGGS writer | `diggs_writer.py` | deterministic, with two gates |
| Scoring one log | `log_scoring.py` | the grid alone, then the reader |

## The record (`model.py`)

`ReportRecord` is what the ingest is **for**; the summary page, the library page
and the DIGGS file are three exports of it. Three rules hold it together.

**A number keeps the unit it was printed in.** A log that prints 21.5 ft is
`Quantity(21.5, "ft")`, and a `Quantity` cannot be built without a unit.
`to_si()` converts once, in the writer that needs SI, so a reviewer setting the
record beside the page sees the page's own numbers.

**Every value-bearing field carries `Provenance`** — the page, the box, and how
it was read: `text`, `di`, `ocr`, `grid`, `vision` or `derived`. A reviewer can
go to the page; a QA pass can ask which values rest on vision alone.

**What could not be read is recorded, not guessed.** An optional field stays
`None`, a `QAEntry` says what was skipped and why, and
`Investigation.units_known` is `False` when no depth unit was printed anywhere.

`LabTest` keeps a free `result` dict until WP3 types it per kind,
`NarrativeFacts` names the owner's two query schemas for WP4 and `CalcEntry` is
WP5. They are in the schema now so a consumer written against it keeps working
as the stubs fill in. `record_json_schema()` exports the whole thing.

## The log reader (`log_reader.py`)

`read_log(doc, item_pages, engine, budget=6)` turns one log — continuation
sheets included — into one `Investigation`.

**Geometry says where, the model says what.** `log_grid` has already found the
columns, fitted the depth ruler and placed every line of text at a (column,
depth) with its box. Those rows are the primary source. The page image at about
110 dpi is for resolving only what the rows leave ambiguous: sample symbols,
water symbols, refusal notation, stacked drives, which of two numbers is the N
value. Every value the picture settled comes back in `changes`, because rows can
be gone back to and a look cannot.

**The depth gate is the point.** A ruler is a linear map from y to depth, so it
says what depth sits at the top edge of the paper and what sits at the bottom;
everything printed on that sheet is between them and nothing else is. A depth
outside that window is refused into `unresolved` rather than accepted. A log
whose pages have no ruler carries no depths at all — the header fields still
come back, because the page still says whose log it is. Total depth is the one
exception and is not gated: on sheet 1 of 3 it names the bottom of the whole
hole, below that sheet's own paper.

**Nothing is computed.** `n` is recorded only where the log prints it; where it
prints only the drives, `n` stays `None`. Half the templates do one and half the
other, and adding the second and third drives of a four-drive rock core would
invent a number. A refusal stays the string the log printed.

**The budget** is one call per log, a second only when the reader itself says it
has pages left, and six at the most.

## The DIGGS writer (`diggs_writer.py`)

`write_diggs(record_or_investigations, project) -> str` produces DIGGS 2.6 XML.
`diggs_schema_gate(xml)` validates it against the schema pydiggs bundles;
`diggs_roundtrip_gate(xml, investigations)` reads it back with the app's own
`parse_diggs` and compares every value. Neither gate alone is enough: a file can
be XSD-valid and wrong, and a file the parser likes may not be DIGGS.

**The 2.6 schema is not the shape the app's older fixtures use**, and this was
checked before anything was written — the app's own fixture fails the schema on
its second line. In real DIGGS the test procedures (`DrivenPenetrationTest`,
`AtterbergLimitsTest`, `WaterContentTest`, `LabDensityTest`, `ParticleSizeTest`,
`UnconfinedCompressiveStrengthTest`, `PocketPenetrometerTest`) are in the
`.../2.6/geotechnical` namespace and carry no result; the value lives in
`Test/outcome/TestResult/results/ResultSet`, named by a `propertyClass` from the
DIGGS dictionary; a depth is a position along the hole's own linear reference
system; lithology is an `observation/LithologySystem` at the document root; and
there is no `WaterLevelObservation` or `MoistureContent` element at all — water
is the borehole's own `waterStrike`. So the writer emits real DIGGS and
`subsurface_characterization/diggs26.py` was added to read it, additively, after
the flat readers. Every existing fixture reads exactly as before.

The element map, record field to DIGGS:

| Record | DIGGS 2.6 |
|---|---|
| `Project` | `project/Project` (`gml:name`, `gml:identifier`) |
| `Investigation` (boring) | `samplingFeature/Borehole` |
| `Investigation` (test pit) | `samplingFeature/TrialPit` |
| `x`, `y`, `elevation` | `referencePoint/PointLocation/gml:pos` |
| `total_depth` | `totalMeasuredDepth` (uom m) |
| `date_started`, `date_finished` | `whenConstructed/TimeInterval` |
| `drilling.method`, `.equipment` | `constructionMethod/BoreholeConstructionMethod` |
| `fields` (anything else printed) | `otherSamplingFeatureProperty/Parameter` |
| `Sample` | `samplingActivity/SamplingActivity` + `sample/Sample` |
| `Sample.top`, `.bottom` | `samplingLocation/LinearExtent/gml:posList` |
| `Sample.recovery` | `totalSampleRecoveryLength` |
| `Sample.recovery_percent` | `otherSamplingActivityProperty/Parameter` — DIGGS has only a length |
| `Sample.rqd_percent` | `samplingActivityRQD` (uom %) |
| `Layer` | `observation/LithologySystem/lithologyObservation/LithologyObservation` |
| `Layer.uscs`, `.description` | `Lithology/classificationCode`, `/lithDescription` |
| `SPT.n` | `ResultSet` `propertyClass` **n_value** |
| `SPT.blows` (no printed N) | `ResultSet` `propertyClass` **blow_count** |
| `SPT.blows`, `.refusal` | `DrivenPenetrationTest/driveSet/DriveSet` (`blowCount`, `penetration`) |
| `drilling.hammer_type`, `.hammer_energy_ratio` | `hammerType`, `hammerEfficiency` |
| `WaterLevel` | `Borehole/waterStrike/WaterStrike` |
| `Sample.water_content` | **water_content_natural**, `WaterContentTest` |
| `Sample.dry_unit_weight` | **dry_density**, `LabDensityTest` |
| `Sample.liquid_limit`, `.plastic_limit`, `.plasticity_index` | **liquid_limit**, **plastic_limit**, **plasticity_index**, `AtterbergLimitsTest` |
| `Sample.fines_percent` | **percent_fines**, `ParticleSizeTest` |
| `Sample.qu` | **compressive_strength_unconfined**, `UnconfinedCompressiveStrengthTest` |
| `Sample.pocket_pen` | **compressive_strength_unconfined**, `PocketPenetrometerTest` |

SI once, here, with a `uom` on every measure. A coordinate gets nine decimals,
because a latitude to four is eleven metres. A unit not in the record's
conversion table is **not** written with a guessed one: it is left out and named
in `DiggsWriteNotes.skipped`, which the round-trip gate then does not look for.
A layer whose base the sheet never printed is written as a contact and named the
same way — DIGGS can carry it, the app's `LithologyInterval` holds intervals
only.

**The gate that matters is deterministic and needs no model.** All fifteen
hand-truthed logs convert to records, write, validate and round-trip:
`module_work/report_ingest_harness/tests/test_diggs_truth.py` (skipped where the
gitignored corpus is not). `report_ingest/tests/test_diggs_writer.py` walks the
same path on a synthetic investigation in CI.

**The numbers come from the cluster.** This app runs in Funhouse against
OpenAI models through the Prompter API, so a score measured on any other model
measures a model that will never do the work. `PrompterEngine` is the engine
that counts and `cluster_scoring.score_on_cluster` is how a run is made.
`ClaudeEngine` stays in the package as a development engine — it is what the
passes were built and debugged against — but its numbers are a checkpoint, not
a result.

## Why the two label passes exist

The rules label a page from what that page prints about itself. Measured on the
corpus, they reach 0.91 accuracy across the fourteen reports they were built on
and 0.79 on the next report anyone opens. The gap is structural, not a missing
rule: a page only makes sense in the context of the rest of the report. An
appendix tab that says "LABORATORY TEST DATA" is right about most of its
appendix and wrong about the summary table, the legend sheet and the stray
calculation bound into it, and no amount of per-page cleverness fixes that.

So both passes are given everything the report says about **itself** before any
of it is read in detail.

## What each pass reads and returns

### `triage(doc, roles, outline, engine=...) -> DocumentProfile`

One call. It is given:

- the per-page ledger, one line a page (`planlens.document.roles.page_ledger`);
- the outline the document prints: contents, lists of figures, tables and
  appendices, every divider's text, figure captions, section headings;
- the front matter verbatim, through the end of the contents;
- the first contact sheet as a picture;
- **the facts Python already counted** — the scanned fraction, the
  unreliable-text fraction, where the printed page numbering restarts, the page
  kinds and the rule labels.

That last point is the rule of the module. A model asked to count 455 ledger
lines will be nearly right, which is the worst thing a scorecard field can be.
Those fields are measured by `document_facts()` and copied into the profile;
`TriageFindings` — the schema the model is held to — does not contain them.

It returns a `DocumentProfile`: `document_type` (the owner's enumeration),
`bound_together`, `has_narrative` / `has_logs` / `has_lab` / `has_calcs`,
`languages`, `toc_agreement`, `anomalies`, `rationale`, the measured fractions,
and a `workflow` in `standard`, `appendix_only`, `partial`, `multi_document`,
`scanned`, `needs_person`. The workflow chooses the readers that follow, so an
odd file is flagged rather than forced down the standard path.

### `review_labels(doc, roles, outline, profile, budget, engine=...) -> Review`

An agent loop. Its tools are `read_page(page)` (the toolkit's text for one
page), `render_page(page)` (the page as a picture at 80 dpi),
`contact_sheet(start_page)` (48 thumbnails, each labelled with its page index
and kind) and `outline()`.

Its brief carries the label vocabulary with one-line definitions, the ledger,
the outline, the triage profile, and **the rules' own weak spots as the list to
check first** — every page whose label came from an appendix tab rather than the
page, every page the tab left ambiguous, every page guessed from its shape,
every page with no readable text, and every page whose label contradicts both
its neighbours. A document with no appendix tab anywhere is said so in the first
line, because then nothing inherited and every label is weak.

**Budget:** `max(60, 0.25 × pages)` tool calls. When it is gone, further calls
come back as errors telling the model to report what it has, and
`Review.stopped_on_budget` records it. `max_model_calls` (default 60) stops the
loop itself.

It returns a `Review`. The model supplies `changes` (page, from, to, a reason of
25 words or fewer, and which tool showed it), `structure` (section or appendix
title → the page range it actually occupies) and `unresolved` (pages it could
not settle, with why).

`final_labels` is **not** asked of the model: it is the rules' labels with the
accepted changes applied, in Python. A model re-emitting 455 page-to-label pairs
can drop a page or shift a number silently, and the scorecard would then be
scoring the slip rather than the judgement. Four kinds of change are refused and
recorded in `rejected_changes` instead of applied: a page that does not exist, a
label outside the vocabulary, a second change to a page already changed, and a
change to the label the page already has. A change's `from` is always overwritten
with what the rules really said, so a model that misremembers is visible.

## Running it on the cluster

This is the run that produces the real numbers, and the owner makes it.

**The reports stay where they already are.** They do not have to be renamed,
copied or re-uploaded: point `reports_dir` at the folder that holds them under
the names their authors gave them, and the manifest's source-file column is
what turns each file name into an ID. Three small private files travel with
the run and can sit anywhere the cluster can read:

| File | What it is | Without it |
|---|---|---|
| `MANIFEST.md` | the corpus manifest; its `file` column names each report | IDs cannot be resolved at all, unless the PDFs are already named `R01.pdf` … |
| `trial_pages_working_r2.xlsx` | the hand page labels | the in-sample set runs and produces profiles, but scores nothing |
| `oos_labels.json` | the lead's out-of-sample labels | the two out-of-sample sets run and score nothing |

The Azure Document Intelligence results are read in **either** form —
`<ID>.json.gz` or the uncompressed `DI_data_<original stem>.json` that
Funhouse wrote — from whatever folder `di_dir` names. Without them the scanned
pages are read from their own text layer alone.

```python
# 1. From PyPI through Nexus. planlens 0.5.0 arrives with it.
%pip install "geotech-staff-engineer==5.19.0"
dbutils.library.restartPython()
```

```python
# 2. One cell. fh_prompter is the object you already have.
from report_ingest.cluster_scoring import score_on_cluster

results = score_on_cluster(
    reports_dir  = "/Volumes/<your volume>/reports",          # the PDFs, under their own names
    manifest     = "/Volumes/<your volume>/wp1b/MANIFEST.md",
    labels_xlsx  = "/Volumes/<your volume>/wp1b/trial_pages_working_r2.xlsx",
    oos_labels   = "/Volumes/<your volume>/wp1b/oos_labels.json",
    di_dir       = "/Volumes/<your volume>/report_di",
    out_dir      = "/tmp/report_ingest_wp1b",                 # /tmp or a Volume, never /Workspace
    prompter     = fh_prompter,
    model        = "funhouse-gpt-high",     # the label review, on the tier the app runs on
    triage_model = "funhouse-gpt-medium",   # one call over a ledger; a cheaper tier does
    sets         = ("insample", "oos_open", "oos_blind"),
    stages       = ("labels",),             # add "logs" to score the log reader too
    max_reports  = 2,                       # drop this line after the first run
)
```

### Scoring the log reader as well (`stages=("labels", "logs")`)

The `logs` stage runs `log_grid` and then `read_log` over each hand-truthed
log, and scores the result twice: **before**, from the grid's cells alone, and
**after**, from the record the reader built. The grid runs once and is handed
to the reader, so the difference between the two columns is the model and
nothing else. It needs one more private file — the folder of hand-truthed logs
— and an `OPEN.txt` beside them names the reports the rules were allowed to be
tuned on, so the blind figure stays blind.

```python
results = score_on_cluster(
    reports_dir  = "/Volumes/<your volume>/reports",
    manifest     = "/Volumes/<your volume>/wp1b/MANIFEST.md",
    di_dir       = "/Volumes/<your volume>/report_di",
    truth_dir    = "/Volumes/<your volume>/wp2b/truth/logs",   # <ID>_p<page>.json + OPEN.txt
    out_dir      = "/tmp/report_ingest_wp2b",
    prompter     = fh_prompter,
    model        = "funhouse-gpt-high",
    stages       = ("logs",),       # or ("labels", "logs") for both in one run
    log_budget   = 6,               # model calls per log; the reader's ceiling is six
    max_reports  = 2,               # drop this line after the first run
)
```

Metrics, all against the hand truth: **N values exact**; sample depths,
index values and water levels within 0.15 m; layer tops within 0.3 m; USCS
symbols matched; recovery and RQD where the log prints them; header fields
recovered. Depths are compared in metres whatever the log prints, and the
matching rules are the WP2a ones — a sample is an interval, a blow record
counts when its drives stand at that depth, and an N the log never printed
counts as found when the drives that define it are there, because neither the
grid nor the reader does arithmetic by design.

`RESULTS.md` then carries a second half: before and after per metric for the
open set, the blind set and all logs; a per-log line with model calls, what was
left unresolved and how many values came off the picture rather than the rows;
and the cost per log. `logs/<log id>.json` holds the per-log detail and, like
the label runs, makes the stage restartable — a log that already has one is
skipped.

Locally, `module_work/report_ingest_harness/measure_wp2b_logs.py` does the same
scoring against the development engine, and `--grid-only` prints the before
column with no engine, no key and no network at all.

Bring back **`/tmp/report_ingest_wp1b/RESULTS.md`**. That is the whole report,
and it carries IDs, labels, counts and rates only. The per-report runs and the
triage profiles stay on the cluster in `runs/` and `triage/` unless you move
them deliberately, because a change's reason and a triage rationale can name a
firm, a project or a person.

Notes that matter:

- **`out_dir` must be `/tmp` or a Volume.** A `/Workspace` path is refused up
  front rather than discovered at the end, because those writes are
  non-durable and permission-blocked here (`docs/DATABRICKS_INSTALL.md`).
- **It resumes.** Each report writes its run file as it finishes and a later
  call skips any report that already has one. A detached notebook costs the
  reports that had not finished, not the ones that had. Pass `redo=True` to
  start over, or delete one file to redo one report.
- **Start small.** `max_reports=2` on the first run proves the path end to end
  for the price of two reports.
- **A report the folder does not have is named and skipped**, once, before
  anything runs, rather than failing one report at a time deep in the set.
- **No credential is read.** Authentication is whatever `fh_prompter` was built
  with. Nothing here touches an environment variable or a secret scope.
- **Tokens, not dollars.** Funhouse publishes no per-token price for a
  capability tier, so the run reports tokens and you read the spend from
  Funhouse's own budget endpoint for the same window.

## What the Prompter engine can and cannot do

| | Prompter (OpenAI through Funhouse) | Claude API (development) |
|---|---|---|
| Multi-turn tool loop | yes, through `prompter.client` | yes |
| Images | yes, as `image_url` data URIs | yes |
| Structured output | yes, `response_format` with a strict JSON schema | yes |
| Image inside a tool result | **no** — see below | yes |
| Explicit prompt caching | no; the provider caches what it caches | yes, and it is asked for |
| Budget guard on every call | only on single-shot calls | not applicable |

Two of those need explaining because they changed the code.

**An image cannot go in a tool result.** OpenAI's `role="tool"` message takes a
string and nothing else, and the label review's `render_page` and
`contact_sheet` answer with a picture. So an image-bearing result is sent as a
tool message saying the picture follows, and the picture rides in a user
message immediately after the tool messages. The model sees both and the
ordering rules are kept. On the Claude API the picture simply goes in the tool
result, so the same review does slightly different work on the two engines —
worth remembering when comparing their numbers.

**The tool loop bypasses the SDK's budget guard.** Prompter's own `chat()`
sends exactly one system and one user message, which cannot express an
assistant turn with tool calls, so the loop drives `prompter.client` directly
exactly as the app's `NativeToolEngine` does. Single-shot calls with no tools
still go through `chat()`, so triage keeps the guard and the SDK's logging and
only the review is outside it. On a backend where `prompter.client` is `None`
(Grok) there is no loop at all, and the engine says so rather than failing
deep in a run.

## The engine, and where it runs

Neither pass imports an SDK. Both take an `Engine` — one method:

```python
complete(messages, system=..., tools=..., images=..., output_format=...) -> Reply
```

Messages are neutral blocks (`text`, `image`, `tool_use`, `tool_result`,
`opaque`), and the engine translates to and from its provider's shape in both
directions. `opaque` is how a pass carries a thinking block back to the model
unchanged without ever reading it. So the same passes run on the Claude API here
and on the cluster's Prompter later, behind the app's own engine abstraction.

`ClaudeEngine` is the development implementation. **The credential comes from
the environment** — `ANTHROPIC_API_KEY`, or an `ant auth login` profile. It is
never an argument, never logged and never printed; only its presence is checked,
and a missing one fails at construction rather than deep inside a run. The
prompt prefix is cached by default (`auto_cache=True`), which on the checkpoint
turned roughly 1.4 M resent tokens into cache reads at a tenth the price.

`CostMeter` records calls, input and output tokens, cache reads and writes, wall
clock and dollars at the list prices in `MODEL_PRICES` (checked 2026-09-16).
Prices are pinned in the source rather than fetched: a scorecard has to be able
to say what a run cost months later.

## Running the scorecard

```
.venv/Scripts/python -m module_work.report_ingest_harness.measure_wp1b \
    --set checkpoint --note "what changed this round"
.venv/Scripts/python -m module_work.report_ingest_harness.measure_wp1b \
    --set insample --reuse --changes
.venv/Scripts/python -m module_work.report_ingest_harness.measure_wp1b \
    --set oos_blind --no-append
```

Sets are `insample` (the reports with spreadsheet labels), `oos_open` (the
lead's out-of-sample labels on ten reports), `oos_blind` (the other fourteen)
and `checkpoint` (six reports chosen to span short, long, scanned and old).
Running the first three covers all 38 corpus reports exactly once, which is also
how every report gets a triage profile. `--reuse` re-scores the saved runs
instead of calling the model, so tuning the scorecard is free.

Each run writes the full private detail to `raw/checks/wp1b/` and each report's
profile to `raw/checks/triage/`, both gitignored. The ledger gets IDs, labels,
counts and rates only, never a heading, a change's reason or a triage
rationale.

**Every change is graded** against the hand label as `fixed`, `broke`,
`still_wrong`, `disputed` or `unscored`. A review that raises accuracy while
breaking three correct labels has not earned the raise, and a total alone would
hide it.

**The blind set is protected and reported honestly.** `oos_blind` prints one
summary, writes no miss list and no change list, and refuses `--changes`. Two of
its fourteen reports were named in the cost checkpoint and their changes were
read, so the set is reported twice: over all fourteen, and over the twelve
nobody has opened. The second figure is the honest one and the output says so.

**Disputed hand labels.** When the review contradicts a hand label and the lead,
looking at the page, judges the hand label the doubtful one, the entry goes in
`DISPUTED`. The spreadsheet is never edited: a hand label records what a person
decided, and rewriting it to suit a model would destroy the only independent
thing in the measurement. A confirmed dispute drops the page from the before
**and** the after score, so it counts as neither a rule hit nor a review miss,
and the number of dropped pages is printed. It only drops while the review still
says what it said when the dispute was raised; a later run that moves the page
somewhere else is a new answer and is scored.

**Three guards, each from a run that went wrong.** A run in which every report
fails no longer overwrites the last good run's detail file. Every saved run
records a hash of the prompts that made it, and scoring a set whose saved runs
disagree refuses to stand behind the totals — half a set on new prompts and half
on old is a set that never existed. And `--max-report-dollars` (default $5) and
`--max-total-dollars` (default $45) stop the set rather than spend past them,
naming the report they stopped before.

## Tests

```
.venv/Scripts/python -m pytest report_ingest/tests -q
.venv/Scripts/python -m pytest module_work/report_ingest_harness/tests -q
```

Offline and free: no credential, no network, no corpus. `FakeEngine` replays a
script of turns in the engine's place, so the whole of both label passes and the
log reader runs for real — the prompt each builds, the tool loop, the budget
stop, the rules for applying a change. Running past the end of a script raises,
because a pass that makes one more call than the test expected is the bug the
test exists to catch.

The log reader's tests run over real `log_grid` output on planlens' synthetic
log fixtures, so the brief that is asserted on is the brief a model would
actually be sent, and the refusals are the refusals that would actually happen:
a depth past the ruler, a log with no scale at all, a provenance naming a page
outside this log. The DIGGS writer is tested against the bundled 2.6 schema and
through `parse_diggs` on a synthetic investigation carrying one of everything,
and the log scorer on a synthetic truth where what is checked is the matching
itself.

The harness suite adds the scorecard's own arithmetic, the disputed-label rule,
the prompt fingerprint, and the deterministic fifteen-log DIGGS gate. Its
corpus-dependent tests skip cleanly on a machine without the private data.

## What the measurements have taught the prompts

Kept here because each line is a prompt rule that exists for a measured reason,
and a later editor who does not know why will delete it.

- **The four exploration labels say they may be PLOTTED, not just tabulated,
  and that each belongs to one named exploration.** On the cost checkpoint,
  fifteen of the review's nineteen wrong changes were an exploration's own
  results called `figure`, or one kind of sounding called another. The rules had
  been wrong about those pages and the review was right to move them; it moved
  them next door. A page showing depth against blow count looks like a graph,
  and nothing had said that a graph of one exploration's results is that
  exploration's log.
- **`figure` says it is the report's own numbered figure series and never an
  exploration's results**, for the same reason, from the other side.
- **The review is told to choose among the four logs on what is MEASURED, not on
  how it is drawn**: blows per increment of penetration is a DCP, continuous tip
  and sleeve and pore pressure is a CPT, a logged pit or trench face is a test
  pit, a drilled hole with driven samples and blow counts is a boring.
- **The brief leads with the rules' own weak spots.** The out-of-sample study
  sorted every rule miss into four classes and three are answerable only by
  opening the page, so those pages are named first and the rest of the document
  follows.
- **The deterministic facts are given, never asked.** A model asked to count 455
  ledger lines will be nearly right, which is the worst thing a scorecard field
  can be.

## How this package ships

It ships in the wheel as of **5.19.0** (2026-09-17) as a library: no tool of
the app calls it yet, and nothing in the chat surface changed.

- **`report_ingest*` is in `[tool.setuptools.packages.find]`.** Until that line
  landed, a wheel built from this repo did **not** contain this package, and
  the failure would have appeared on the cluster as an import error rather
  than at build time. Check it after any edit to that list:
  `python -m build --wheel` and look for `report_ingest/` in the wheel.
- **`report_ingest` is in pytest's `testpaths`**, so this suite runs in the
  release gate. The WP0/WP1 harness under `module_work/` is dev-only and is
  deliberately not.
- **`planlens>=0.5`** — the page roles, the printed outline and the per-page
  ledger these passes read are 0.5.0's. planlens 0.5.0 declares exactly the
  dependencies 0.4.0 did, so the pin adds no new package to the cluster.
- `pydantic` and `openpyxl`, both already there: pydantic through the agent
  stack, openpyxl as a hard requirement of `python-ags4`, which this app
  depends on directly. The hand-label reader imports openpyxl lazily anyway,
  so a run that never touches the spreadsheet never needs it.
- `anthropic` is **optional**, and on the cluster not installed at all.
  Nothing outside `ClaudeEngine` imports it, `report_ingest/__init__.py` reaches
  every entry point through a lazy import, and a test spawns a fresh
  interpreter to prove that importing the package pulls in neither `anthropic`
  nor `planlens`.

## How the corpus is read

`corpus.Corpus` takes directories as arguments — the PDFs, the Azure results,
the spreadsheet — because the same corpus is a gitignored folder in this repo
during development and a Volume on the cluster during a real run. The WP0
harness (`module_work/report_ingest_harness/`) is now that loader bound to the
repo's own paths, so there is one implementation and the harness's own tests
check it.

It reads **two layouts**, because the repo's copy was renamed and the
cluster's was not. `R01.pdf` … is one; the reports under their original names
plus a manifest is the other, and an ID is resolved through the manifest's
source-file column — the path as written, then without its leading folder
(`Reports_PDF/`), then on the base name case-insensitively, then on a
punctuation- and accent-insensitive form of the stem. DI results are read as
`<ID>.json.gz` or as `DI_data_<original stem>.json`, gzip preferred where both
exist and the file sniffed rather than trusted from its suffix. Both
resolutions and both DI forms are pinned by offline tests on synthetic files
(`tests/test_corpus_paths.py`), because the cluster run is the one place this
code has to work first time. `scoring.py` holds the rates, the sets and the change verdicts for the
same reason: two copies of that arithmetic would drift, and the second copy's
numbers would be the ones nobody checked.
